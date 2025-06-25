# from typing import Union, overload
#
# from veropt.optimiser.constructors import SingleKernelOptions, KernelInputDict
# from veropt.optimiser.model import GPyTorchSingleModel
#
#
# # TODO: Make this work! :))
#
#
# @overload
# def gpytorch_single_model_list(
#         n_variables: int,
#         n_objectives: int,
#         kernels: list[GPyTorchSingleModel] = ...,
#         kernel_settings: None = ...
# ) -> list[GPyTorchSingleModel]: ...
#
#
# @overload
# def gpytorch_single_model_list(
#         n_variables: int,
#         n_objectives: int,
#         kernels: 'SingleKernelOptions' = ...,
#         kernel_settings: Union['KernelInputDict', None] = ...
# ) -> list[GPyTorchSingleModel]: ...
#
#
# @overload
# def gpytorch_single_model_list(
#         n_variables: int,
#         n_objectives: int,
#         kernels: list['SingleKernelOptions'] = ...,
#         kernel_settings: Union[list['KernelInputDict'], None] = ...
# ) -> list[GPyTorchSingleModel]: ...
#
#
# @overload
# def gpytorch_single_model_list(
#         n_variables: int,
#         n_objectives: int,
#         kernels: None = ...,
#         kernel_settings: None = ...
# ) -> list[GPyTorchSingleModel]: ...