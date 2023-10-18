import bluepysnap.circuit_ids_types as test_module


class TestCircuitNodeId:
    def setup_method(self):
        self.test_obj = test_module.CircuitNodeId("pop", 1)

    def test_init(self):
        assert isinstance(self.test_obj, test_module.CircuitNodeId)
        assert isinstance(self.test_obj, tuple)

    def test_accessors(self):
        assert self.test_obj.population == "pop"
        assert self.test_obj.id == 1


class TestCircuitEdgeId:
    def setup_method(self):
        self.test_obj = test_module.CircuitEdgeId("pop", 1)

    def test_init(self):
        assert isinstance(self.test_obj, test_module.CircuitEdgeId)
        assert isinstance(self.test_obj, tuple)

    def test_accessors(self):
        assert self.test_obj.population == "pop"
        assert self.test_obj.id == 1
