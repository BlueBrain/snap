import pytest

from bluepysnap.exceptions import BluepySnapError
from bluepysnap.sonata_constants import ConstContainer


class Container(ConstContainer):
    VAR1 = "var1"


class SubContainer(Container):
    VAR2 = "var2"


class SubSubContainer1(SubContainer):
    VAR3 = "var3"


class SubSubContainer2(SubContainer):
    VAR4 = "var4"


class ComplexContainer(SubSubContainer1, SubSubContainer2):
    VAR5 = "var5"


class FailingContainer(SubContainer, int):
    VAR6 = "var6"


def test_list_keys():
    assert SubSubContainer1.key_set() == {"VAR1", "VAR2", "VAR3"}
    assert SubSubContainer2.key_set() == {"VAR1", "VAR2", "VAR4"}
    assert ComplexContainer.key_set() == {"VAR1", "VAR2", "VAR3", "VAR4", "VAR5"}

    with pytest.raises(BluepySnapError):
        FailingContainer.key_set()


def test_get():
    assert SubSubContainer1.get("VAR1") == "var1"
    assert SubSubContainer1.get("VAR3") == "var3"
    with pytest.raises(BluepySnapError):
        SubSubContainer1.get("VAR4")
