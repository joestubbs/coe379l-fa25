Supplement: Pandas Operations For Transforming Data 
====================================================

The Pandas library includes some advanced features that are quite useful 
for data processing and analysis. In this supplement, we provide an overview 
of some of the most useful (and at times, confusing) features. 

Overview
--------
We'll look at four functions in roughly increasing order: ``map``, ``applymap``, ``apply``, 
and ``transform``. At a high level, these functions all work similarly: they accept 
an input function and use it to modify elements in a dataframe. However, each is used 
in slightly different situations. Here is a summary:

* ``map`` -- used to apply a function element-wise to a **Series**. This is useful 
  for very simple transformations, but keep in mind it can only be applied to a Series 
  object (e.g., a column).
* ``applymap`` -- similar to ``map`` in that is applies a function element-wise, but it 
  can only be used with a **DataFrame**. Again, useful for simple transformations. 
* ``apply`` -- used to apply a function along an axis (either rows or columns) in a way 
  that could change the shape of the dataframe. 
* ``transform`` -- Similar to ``apply``, ``transform`` is used to apply a function along 
  an axis 

One thing to keep in mind with all of the above operations is that they take an *input 
function* to apply. This function is often times specified as an anonymous function, i.e., 
using the ``lambda`` keyword syntax. For example, one could express the doubling function
in using lambda syntax as follows:

.. code-block:: python 

    lambda x: x * 2

Let's look at each operation in detail with some examples. 

Map
---
Possibly the simplest function, ``map`` applies a function element-wise, i.e., to every element 
in a Series. Thus, we cannot apply ``map`` to an entire dataframe, but we can apply it 
to a column. For example, given the following dataframe:

.. code-block:: python 

    df = pd.DataFrame({
      "A": [1, 2, 3],
      "B": [10, 20, 30]
    })

We could apply ``map`` to the ``A`` column, passing the doubling function as follows:

.. code-block:: python 

  df["A"].map(lambda x: x * 2)

This would multiply every element of the Series by 2. Note that this doesn't actually 
modify the original ``df``, it returns a new Series object. So if we want to update the 
``"A"`` column, we need to save it back to the dataframe, like so:

.. code-block:: python 

  >>> df["A"] = df["A"].map(lambda x: x * 2)
  >>> df
        A 	B
  0 	2 	10
  1 	4 	20
  2 	6 	30

Applymap
--------
The ``applymap`` function works the same as ``map`` but for an entire dataframe. Therefore, 
given the dataframe from before: 

.. code-block:: python 

    df = pd.DataFrame({
      "A": [1, 2, 3],
      "B": [10, 20, 30]
    })

We can apply the doubling function to every element in the dataframe as follows:

.. code-block:: python 

    >>> df.applymap(lambda x: x * 2)

But as before, this does not modify the ``df`` object; it returns a new dataframe instead. 
So, if we want to modify the ``df``, we need to save the result back to it, like so: 


.. code-block:: python 

  >>> df = df.applymap(lambda x: x * 2)  
  >>> df
     	A 	B
  0 	2 	20
  1 	4 	40
  2 	6 	60

Apply 
-----

The ``apply`` function is used to apply a function to a specific axis, either columns 
(``axis=0``) or rows (``axis=1``). In this way, it works similarly to ``transform`` 
which we will talk about last, but the key thing to keep in mind that ``apply`` may 
change the shape of the dataframe! Let's see some examples. 

At first, this may need seem like a big deal, since if we have numeric data and we apply 
our doubling function, the result of ``apply`` is the same as that of ``applymap``, whether 
we use rows or columns: 

.. code-block:: python 

    df = pd.DataFrame({
      "A": [1, 2, 3],
      "B": [10, 20, 30],
      "C": [5, 15, 20],
      "D": [100, 150, 170]
    })    

.. code-block:: python 

    >>> df.apply(lambda x: x * 2, axis=0)
    	A 	B 	C 	D
    0 	2 	20 	10 	200
    1 	4 	40 	30 	300
    2 	6 	60 	40 	340

.. code-block:: python 
    
    # gives the same result! 
    >>> df.apply(lambda x: x * 2, axis=1)
    	A 	B 	C 	D
    0 	2 	20 	10 	200
    1 	4 	40 	30 	300
    2 	6 	60 	40 	340

However, let's see what happens when we pass the ``sum`` function:

.. code-block:: python 

    >>> df.apply(sum, axis=0)
    A      6
    B     60
    C     40
    D    420
    dtype: int64

The shape of the dataframe has been changed entirely, as it has collapsed all rows into 
a single value (the sum). And of course, if we change the axis, we get a very different 
result: 

.. code-block:: python 

    >>> df.apply(sum, axis=1)
    0    116
    1    187
    2    223
    dtype: int64

In this case, it summed the rows, as expected. Keep in mind that none of these changed 
the actual contents of the ``df`` object. 


Transform 
----------
Finally, let's look at ``transform``, which applies a function to either the columns (``axis=0``)
or the rows (``axis=1``), just like with ``apply``, but this time, it must preserve the 
original shape of the dataframe. You almost always use ``transform`` in conjunction with a 
``groupby``. A standard use of ``transform`` is to fill in missing values. 

Note that ``transform`` gives the exact same result as ``apply`` and ``applymap`` when 
passed the doubling function: 

.. code-block:: python 

    >>> df.transform(lambda x: x * 2, axis=0)
    	A 	B 	C 	D
    0 	2 	20 	10 	200
    1 	4 	40 	30 	300
    2 	6 	60 	40 	340

But let's look at a slightly more complicated dataframe; suppose we have: 

.. code-block:: python 

    cars = pd.DataFrame({
      "brand": ["Toyota", "Toyota", "Tesla", "Tesla"],
      "price": [20000, 25000, 80000, 90000]
    })

If we use ``groupby`` to collect elements by brand and then access the ``price`` column, 
we can then use ``transform`` to apply a function like ``sum``, and the result is a 
new Series of the same length:

.. code-block:: python 

    >>> cars.groupby("brand")["price"].transform(sum)
    0     45000
    1     45000
    2    170000
    3    170000
    Name: price, dtype: int64

The sums of each column are repeated for every value of the same brand. Note that this 
behavior differs from that of ``apply``:

.. code-block:: python 

    >>> cars.groupby("brand")["price"].apply(sum)
    brand
    Tesla     170000
    Toyota     45000
    Name: price, dtype: int64