

def isiterable(obj):
    """
    Checks if object is iterable by seeing if it has the '__iter__' attribute.

    Parameters
    ----------
    obj: *
        Any Python object to test.

    Returns
    -------
    bool:
        Obj is an iterable or not.

    Examples
    -------
    >>> isiterable(1)
    False
    >>> isiterable(100.10930)
    False
    >>> isiterable([1])
    True
    >>> isiterable('1')
    True
    >>> isiterable([1,2,3])
    True
    >>> isiterable(range(10))
    True
    >>> isiterable({1:1,2:2})
    True
    
    """
    
    return( hasattr(obj,'__iter__') )
