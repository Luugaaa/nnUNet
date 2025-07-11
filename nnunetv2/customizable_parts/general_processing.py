from nnunetv2.preprocessing.normalization.default_normalization_schemes import ZScoreNormalization, CTNormalization, NoNormalization, RescaleTo01Normalization, RGBTo01Normalization
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape, resample_data_or_seg_to_spacing, resample_data_or_seg

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient, NibabelIO
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.imageio.tif_reader_writer import Tiff3DIO

normalizer_classes = {
    "ZScoreNormalization": ZScoreNormalization,
    "CTNormalization": CTNormalization,
    "NoNormalization": NoNormalization,
    "RescaleTo01Normalization": RescaleTo01Normalization,
    "RGBTo01Normalization": RGBTo01Normalization,
}

resampling_fn_list = {
    "resample_data_or_seg_to_shape": resample_data_or_seg_to_shape,
    "resample_data_or_seg_to_spacing": resample_data_or_seg_to_spacing,
    "resample_data_or_seg": resample_data_or_seg,
}

image_io_classes = {
    "NibabelIOWithReorient": NibabelIOWithReorient,
    "NibabelIO": NibabelIO,
    "SimpleITKIO": SimpleITKIO,
    "Tiff3DIO": Tiff3DIO
}