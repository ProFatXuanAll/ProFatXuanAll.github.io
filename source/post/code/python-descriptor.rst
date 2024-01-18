==================================
Python Descriptor 運作邏輯
==================================

.. ====================================================================================================================
.. Setup SEO.
.. ====================================================================================================================

.. meta::
  :description:
    Python Descriptor 運作邏輯
  :keywords:
    Descriptor,
    Python

.. ====================================================================================================================
.. Setup front matter.
.. ====================================================================================================================

.. article-info::
  :author: ProFatXuanAll
  :date: 2024-01-17
  :class-container: sd-p-2 sd-outline-muted sd-rounded-1

.. ====================================================================================================================
.. Create visible tags from SEO keywords.
.. ====================================================================================================================

:bdg-secondary:`Descriptor`
:bdg-secondary:`Python`

基本語法
========

請見 `python 官方教學 <python-descriptor-how-to_>`_\，以下 descriptor 範例程式碼改寫自官方教學。

.. code-block:: python
  :linenos:

  class Printer:

    def __set_name__(self, owner, name):
      self.public_name = name
      self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
      value = getattr(obj, self.private_name)
      print('Accessing %r giving %r', self.public_name, value)
      return value

    def __set__(self, obj, value):
      print('Updating %r to %r', self.public_name, value)
      setattr(obj, self.private_name, value)

  class Person:

    name = Printer() # 創造 descriptor instance `name` 同時觸發 `name.__set_name__(Person, 'name')`
    age = Printer()  # 創造 descriptor instance `age`  同時觸發 `age.__set_name__(Person, 'age')`

    def __init__(self, name, age):
      self.name = name # 使用 instance `person` 與 value `name` 觸發 `name.__set__(person, name)`
      self.age = age   # 使用 instance `person` 與 value `age`  觸發 `age.__set__(person, age)`

    def birthday(self):
      # 1. 使用 instance `person` 觸發 `age.__get__(person, Person)`
      # 2. 使用 instance `person` 與 value `age + 1`  觸發 `age.__set__(person, age + 1)`
      self.age += 1

名詞定義
========

data descriptor
  有定義 ``__set__`` **or** ``__delete__`` methods 的 descriptor。
  請注意連接詞 **or**\，代表有定義一個就滿足條件，但實務上幾乎都會定義 ``__set__``。
  請注意此定義不考慮是否有定義 ``__get__`` method，但實務上通常會配合 ``__set__`` 一起定義。

non-data descriptor
  只定義 ``__get__`` method 的 descriptor。

read-only data descriptor
  同時定義 ``__get__`` and ``__set__`` methods，但 ``__set__`` 的實做內容會觸發 ``raise AttributeError``\。
  由於 calling ``__set__`` 只會觸發錯誤，因此為 read-only descriptor。

.. _code.python-descriptor.instance:

Access descriptor through instance attribute
============================================

令 ``D`` 為任意 descriptor class，令 ``a`` 為任意 class，令 ``a`` 為 ``A`` 的 instance。
當程式碼嘗試透過 ``a.x`` 存取 attribute ``x`` 時，根據 `python 官方教學 <python-descriptor-how-to_>`_ 將會觸發以下的查詢邏輯嘗試取得 attribute ``x``。

0. 列舉 ``type(a).__mro__`` 尋找 method ``__getattribute__``
1. 如果步驟 0 找到的 ``__getattribute__ is not object.__getattribute__``\，則執行找到的 ``__getattribute__``\，觸發 ``a.__getattribute__('x')`` 並將回傳值作為 ``a.x``
2. 如果步驟 0 找到的 ``__getattribute__ is object.__getattribute__``\，則執行 python 內建的 attribute lookup logic，定義於後續步驟中
3. 尋找包含 ``x`` 的 base class ``B``：

   a. 列舉 ``type(a).__mro__``\，令列舉的變數為 ``cls``，檢查 ``'x' in cls.__dict__`` 是否為真
   b. 如果是，則令 ``B = cls``\，並結束列舉，同時回傳 ``B.__dict__['x']``
   c. 如果結束列舉且所有檢查都為否，則令 ``B = object``

4. 如果步驟 3 有回傳值 ``d``\，且檢查後發現 ``d`` 為 data descriptor，則觸發 ``d.__get__(a, B)`` 並將回傳值作為 ``a.x``
5. 如果步驟 3 有回傳值 ``d``\，且檢查後發現 ``d`` 不為 data descriptor，且檢查後發現 ``'x' in a.__dict__`` 為真，則將 ``a.__dict__['x']`` 作為 ``a.x``
6. 如果步驟 3 有回傳值 ``d``\，且檢查後發現 ``d`` 為 non-data descriptor，且檢查後發現 ``'x' in a.__dict__`` 為否，則觸發 ``d.__get__(a, B)`` 並將回傳值作為 ``a.x``
7. 如果步驟 3 有回傳值 ``d``\，且檢查後發現 ``d`` 不為 descriptor，且檢查後發現 ``'x' in a.__dict__`` 為否，則將 ``d`` 作為 ``a.x``
8. 前述步驟皆不成立，則觸發 ``AttributeError('x')``
9. 當觸發 ``AttributeError('x')`` 時，列舉 ``type(a).__mro__`` 尋找 method ``__getattr__``
10. 如果步驟 9 找到的 ``__getattr__ is not object.__getattr__``\，則執行找到的 ``__getattr__``\，觸發 ``a.__getattr__('x')`` 並將回傳值作為 ``a.x``
11. 如果步驟 9 找到的 ``__getattr__ is object.__getattr__``\，則觸發 ``AttributeError('x')``

以下範例程式碼來自官方教學，透過 python 實做 instance attribute access。

.. code-block:: python
  :linenos:

  def mock__find_name_in_mro(cls, name, default):
    "Emulate _PyType_Lookup() in Objects/typeobject.c"
    for base in cls.__mro__:
      if name in vars(base):
        return vars(base)[name]
    return default

  def mock__getattribute__(obj, name):
    "Emulate PyObject_GenericGetAttr() in Objects/object.c"
    null = object()
    objtype = type(obj)
    cls_var = mock__find_name_in_mro(objtype, name, null)
    descr_get = getattr(type(cls_var), '__get__', null)
    if descr_get is not null:
      if hasattr(type(cls_var), '__set__') or hasattr(type(cls_var), '__delete__'):
        return descr_get(cls_var, obj, objtype)        # data descriptor
    if hasattr(obj, '__dict__') and name in vars(obj):
      return vars(obj)[name]                           # instance variable
    if descr_get is not null:
      return descr_get(cls_var, obj, objtype)          # non-data descriptor
    if cls_var is not null:
      return cls_var                                   # class variable
    raise AttributeError(name)

  def mock__getattr__(obj, name):
    "Emulate slot_tp_getattr_hook() in Objects/typeobject.c"
    try:
      return obj.__getattribute__(name)
    except AttributeError:
      if not hasattr(type(obj), '__getattr__'):
        raise
    return type(obj).__getattr__(obj, name)            # __getattr__

補充說明：

- 區分步驟 4 跟 6 的原因是為了保障 ``d.__set__`` 有先被執行過才能觸發 ``d.__get__`` 的邏輯
- 步驟 8 跟 11 的 ``AttributeError('x')`` 是同一個 error 只是在不同 context 觸發，觸發的 context 分別為 ``__getattribute__`` 與 ``__getattr__``

.. _code.python-descriptor.class:

Access descriptor through class attribute
=========================================

因為 class 本身是 instance of ``type``\，所以可以使用\ :ref:`相同邏輯 <code.python-descriptor.instance>`\執行 attribute lookup。
如果透過 ``A.x`` 存取 attribute ``x`` 時，步驟 4 與 6 會改成呼叫 ``d.__get__(A, None)``\。

.. _`python-descriptor-how-to`: https://docs.python.org/3/howto/descriptor.html
