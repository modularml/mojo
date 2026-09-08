:title: max.driver
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.driver
==========

.. automodule:: max.driver
   :no-members:

.. currentmodule:: max.driver

Devices
-------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Accelerator
   CompletionFlag
   CPU
   Device
   DeviceEvent
   DeviceQueue
   DeviceSpec

Buffers
-------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Buffer
   DevicePinnedBuffer
   DLPackArray
   Usage

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   batch_inplace_copy
   load_max_buffer

Device discovery
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   accelerator_api
   accelerator_architecture_name
   accelerator_count
   devices_exist
   enable_all_peer_access
   load_devices
   scan_available_devices

Launch tracing
--------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   LaunchTraceEntry

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   begin_launch_trace
   launch_trace
   take_launch_trace

Virtual devices
---------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   calculate_virtual_device_count
   calculate_virtual_device_count_from_cli
   get_virtual_cpu_target
   get_virtual_device_api
   get_virtual_device_count
   get_virtual_device_target_arch
   is_virtual_device_mode
   set_virtual_cpu_target
   set_virtual_device_api
   set_virtual_device_count
   set_virtual_device_target_arch
