#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os

"""
PLEASE READ documentation/setting_up_paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

nnUNet_results = os.environ.get('nnUNet_results')

if nnUNet_results is None:
    raise ValueError("Please set the environment variable nnUNet_results. Example: export nnUNet_results='inference_results'")
