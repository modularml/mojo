## Expected Optimizations

The author of this proposal intuitively finds that several optimizations
beyond existing Actor Model implementations can be achieved by using Mojo as 
the primary implementation language. This section outlines those 
expected optimizations.

## Message Serialization Optimization
Actors can and should be distributed across processing nodes. Consequently, the
messages they pass need to be serialized for network transmission. In extremely
busy actor systems, serialization can be a significant chunk of the processing.

To use hyperbole to make a point, consider:
* A message `AddValue(value: Int)` sent to an actor
* The processing for that message being just the addition of `value` to
  some integer value in the actor's state

In such a situation, the processing time of the resulting behavior can be
counted in the low number of nanoseconds (update a memory location), but
if the actor is remote, serialization of the message involves codifying
both the name of the message (`AddValue`) and its parameters (`value`).
Consequently, the serialization cost considerably outweighs
the message processing cost. While traditional methods of serialization are
adequate, this area is ripe for a number of optimizations, including:

* Avoiding serialization altogether if recipient actor is on the same memory
  system and passing the message by immutable reference instead. This 
  involves factoring in the origin and destination of the message as part of
  message sending.
* Memoization of the message name and values (for repeated message construction). This
  just trades compute time for memory so needs to be balanced for very-frequently
  sent and small messages. 
* Templatization of the message to pre-compute the basic layout of the serialized
  form with "fill-in-the-blanks" areas for the message parameters whose sizes
  should be static because memory size of types is bounded and known by the compiler.
* Bitwise packing of the message should that prove optimal (the packing and
  unpacking speed may not be worth the reduced transmission duration)
* Parallelization for serialization when vectors and arrays are used in the
  message.

## Message Transmission Acceleration

Because message passing is asynchronous and non-blocking, it should be possible to 
accelerate it with an MTPU (Message Transmission Processing Unit)
that is specifically designed to do massive numbers of concurrent message transmissions. 
It could also handle serdes and be combined  with similar accelerations such as NPUs
and IPUs for optimization of network transmission. If such an MPPU was invented, Mojo
should be able to use it to accelerate message passing in hardware. As a fallback 
alternative, there should be a way to utilize FPGA or GPU capabilities to accelerate
message transmission. 

> TODO: The author is not sufficiently versed in hardware design to 
> expand on this further, and those with such familiarity are welcome to
> expand on this idea here. 


## Distribution & Network Optimization

Actor systems can, and should, be distributed across many nodes for redundancy, 
reliability, scalability, and other reasons.  This means
that network acceleration factors into overall actor system design. While NICs, 
routers and switches have all had ASIC designs for many years, the next
generation of hardware for this kind of acceleration is getting more
sophisticated. The author's expectation is that such processors (NPU, IPU)
can be "conditioned" and "configured" programmatically to support distributed
actor systems over many nodes efficiently, and possibly dynamically, based on
load change patterns.

It may even be possible that FPGA like designs in the future could cooperatively
adapt network processing to suit a specific Actor Systems message throughput.  

References:
* [Network Processor](https://en.wikipedia.org/wiki/Network_processor)
* [Content Processor](https://en.wikipedia.org/wiki/Content_processor)
* [Network Processing Unit](https://www.sciencedirect.com/topics/computer-science/network-processing-unit)
* [What is an IPU?](https://www.networkcomputing.com/data-centers/what-ipu-infrastructure-processing-unit-and-how-does-it-work)

> TODO: The author is not sufficiently versed in hardware design to
> expand on this further, and those with such familiarity are welcome to
> expand on this idea here. 


