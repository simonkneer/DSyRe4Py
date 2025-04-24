def prod(val) : 
    """this is just a function to return the product 
    of all elements in a tuple. This is mainly used for 
    reshaping.
    Args:
        val (tuple): input tuple to form the product

    Returns:
        int: product of tuple entries
    """
    val = list(val)
    res = 1
    for ele in val: 
        res *= ele 
    return res  
