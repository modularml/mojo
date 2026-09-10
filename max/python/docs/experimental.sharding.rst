:title: max.experimental.sharding
:type: module
:lang: python
:wrapper_class: rst-module-autosummary

max.experimental.sharding
=========================

.. automodule:: max.experimental.sharding
   :no-members:

.. currentmodule:: max.experimental.sharding

Device mesh
-----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   DeviceMesh

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   get_active_mesh
   mesh_context

Placements
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Partial
   Placement
   ReduceOp
   Replicated
   Sharded
   Collective

Layouts
-------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   BufferLayout
   TensorLayout

Per-op decisions
----------------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   Action
   ActionSet
   AxisAssignment
   PerShard

Pickers
-------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   GreedyReshard
   NoReshard
   PartialsOnly
   ReshardBehavior
   Solver

Exceptions
----------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/class.rst

   ShardingError

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/function.rst

   as_device_mapping
   as_layout
   build_action_set
   force_replicated_action_set
   mode
