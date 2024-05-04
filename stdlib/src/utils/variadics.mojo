from sys.intrinsics import _mlirtype_is_eq


alias _AnyTypeMetaType = __mlir_type[`!lit.anytrait<`, AnyType, `>`]


@always_inline("nodebug")
fn variadic_size[
    ElementTrait: _AnyTypeMetaType,
    *ElementTypes: ElementTrait,
]() -> Int:
    """Get the length of a variadic list of types.

    Parameters:
        ElementTrait: The trait that each element of the variadic conforms to.
        ElementTypes: The list of types held by the argument pack.

    Returns:
        The length of the variadic.
    """

    @parameter
    fn variadic_size(
        x: __mlir_type[`!kgen.variadic<`, ElementTrait, `>`]
    ) -> Int:
        return __mlir_op.`pop.variadic.size`(x)

    alias result = variadic_size(ElementTypes)
    return result


fn all_collection_types_eq[
    T: CollectionElement, *Ts: CollectionElement
]() -> Bool:
    """ """
    alias var_size = variadic_size[CollectionElement, Ts]()
    var eq = True

    @parameter
    fn _item_eq[i: Int]():
        eq = eq and _mlirtype_is_eq[T, Ts[i]]()

    unroll[_item_eq, var_size]()

    return eq
