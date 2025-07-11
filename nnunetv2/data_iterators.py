import multiprocessing
import queue
from torch.multiprocessing import Event, Queue, Manager

from time import sleep
from typing import Union, List

import numpy as np
import torch

from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

from nnunetv2.customizable_parts.preprocessor_classes import preprocessor_classes

def preprocess_fromfiles_save_to_queue(list_of_lists: List[List[str]],
                                       list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                       output_filenames_truncated: Union[None, List[str]],
                                       plans_manager: PlansManager,
                                       dataset_json: dict,
                                       configuration_manager: ConfigurationManager,
                                       target_queue: Queue,
                                       done_event: Event,
                                       abort_event: Event,
                                       verbose: bool = False):
    try:
        label_manager = plans_manager.get_label_manager(dataset_json)
        # preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        # preprocessor = DefaultPreprocessor()
        # preprocessor_classes = {
        #     "DefaultPreprocessor": DefaultPreprocessor()
        # }

        preprocessor_name = configuration_manager.preprocessor_name

        if preprocessor_name in preprocessor_classes:
            preprocessor_class = preprocessor_classes[preprocessor_name]
        else:
            raise NotImplementedError(
                f"The preprocessor '{preprocessor_name}' could not be found. "
                f"You are either using a customized preprocessor or one that was not "
                f"implemented in nnU-Net when this inference code was produced."
            )
            
        preprocessor = preprocessor_class()
        for idx in range(len(list_of_lists)):
            data, seg, data_properties = preprocessor.run_case(list_of_lists[idx],
                                                               list_of_segs_from_prev_stage_files[
                                                                   idx] if list_of_segs_from_prev_stage_files is not None else None,
                                                               plans_manager,
                                                               configuration_manager,
                                                               dataset_json)
            if list_of_segs_from_prev_stage_files is not None and list_of_segs_from_prev_stage_files[idx] is not None:
                seg_onehot = convert_labelmap_to_one_hot(seg[0], label_manager.foreground_labels, data.dtype)
                data = np.vstack((data, seg_onehot))

            data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)

            item = {'data': data, 'data_properties': data_properties,
                    'ofile': output_filenames_truncated[idx] if output_filenames_truncated is not None else None}
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass
        done_event.set()
    except Exception as e:
        # print(Exception, e)
        abort_event.set()
        raise e


def preprocessing_iterator_fromfiles(list_of_lists: List[List[str]],
                                     list_of_segs_from_prev_stage_files: Union[None, List[str]],
                                     output_filenames_truncated: Union[None, List[str]],
                                     plans_manager: PlansManager,
                                     dataset_json: dict,
                                     configuration_manager: ConfigurationManager,
                                     num_processes: int,
                                     pin_memory: bool = False,
                                     verbose: bool = False):
    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(list_of_lists), num_processes)
    assert num_processes >= 1
    processes = []
    done_events = []
    target_queues = []
    abort_event = manager.Event()
    for i in range(num_processes):
        event = manager.Event()
        queue = Manager().Queue(maxsize=1)
        pr = context.Process(target=preprocess_fromfiles_save_to_queue,
                     args=(
                         list_of_lists[i::num_processes],
                         list_of_segs_from_prev_stage_files[
                         i::num_processes] if list_of_segs_from_prev_stage_files is not None else None,
                         output_filenames_truncated[
                         i::num_processes] if output_filenames_truncated is not None else None,
                         plans_manager,
                         dataset_json,
                         configuration_manager,
                         queue,
                         event,
                         abort_event,
                         verbose
                     ), daemon=True)
        pr.start()
        target_queues.append(queue)
        done_events.append(event)
        processes.append(pr)

    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        # import IPython;IPython.embed()
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            all_ok = all(
                [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item
    [p.join() for p in processes]

