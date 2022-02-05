import pickle
import random
import geom
from struct import pack,unpack



def evaluate(circuit, g_tables, a_inputs, b_inputs):
    """Evaluate with given inputs.

    Args:
        circuit: A dict containing circuit spec.
        g_tables: The garbled tables.
        a_inputs: A dict mapping Alice's wires to garbled inputs.
        b_inputs: A dict mapping Bob's wires to garbled inputs.

    Returns:
        A dict mapping output wires with their result bit.
    """
    gates = circuit["gates"]  # dict containing circuit gates
    wire_outputs = circuit["out"]  # list of output wires
    wire_inputs = {}  # dict containing Alice and Bob inputs
    evaluation = {}  # dict containing result of evaluation

    wire_inputs.update(a_inputs)
    wire_inputs.update(b_inputs)

    # Iterate over all gates
    for gate in sorted(gates, key=lambda g: g["id"]):
        gate_id, gate_in, msg = gate["id"], gate["in"], None
        # Special case if it's a NOT gate
        if (len(gate_in) < 2) and (gate_in[0] in wire_inputs):
 
            garbled_value_in = wire_inputs[gate_in[0]]
            
            msg = garbled_value_in
        # Else the gate has two input wires (same model)
        elif (gate_in[0] in wire_inputs) and (gate_in[1] in wire_inputs):
            garbled_value_a= wire_inputs[gate_in[0]]
            garbled_value_b= wire_inputs[gate_in[1]]
            c0 = g_tables[gate_id]         
            x, y = geom.prf(garbled_value_a, garbled_value_b)
            p = geom.Point(x, y) 
            dis = geom.dis(p, c0)
            
            div = dis // 2**31

            if div>0:
                dis = dis / (div+1)
            dis = round(dis)
            msg = pack('i',dis)
        if msg:
            wire_inputs[gate_id] = msg

    # After all gates have been evaluated, we populate the dict of results
    for out in wire_outputs:
        evaluation[out] = wire_inputs[out]
    return evaluation


class GarbledGate:
    """A representation of a garbled gate.

    Args:
        gate: A dict containing gate spec.
        garbled_values: A dict mapping each wire to a pair of garbled_values.
    """
    def __init__(self, gate, garbled_values):
        self.garbled_values = garbled_values  # dict of garbled_values
        self.input = gate["in"]  # list of inputs'ID
        self.output = gate["id"]  # ID of output
        self.gate_type = gate["type"]  # Gate type: OR, AND, ...
        self.garbled_table = {}  # The garbled table of the gate
        # A clear representation of the garbled table for debugging purposes
        self.clear_garbled_table = {}
        '''
        # Create the garbled table according to the gate type
        switch = {
            "OR": lambda b1, b2: b1 or b2,
            "AND": lambda b1, b2: b1 and b2,
            "XOR": lambda b1, b2: b1 ^ b2,
            "NOR": lambda b1, b2: not (b1 or b2),
            "NAND": lambda b1, b2: not (b1 and b2),
            "XNOR": lambda b1, b2: not (b1 ^ b2)
        }
        '''
        # NOT gate is a special case since it has only one input
        if (self.gate_type == "NOT"):
            self._gen_garbled_table_not()
        else:
            self._gen_garbled_table()

    def _gen_garbled_table_not(self):
        """Create the garbled table of a NOT gate."""
        inp, out = self.input[0], self.output
        
        # For each entry in the garbled table

        self.garbled_values[out] = (self.garbled_values[inp][1],self.garbled_values[inp][0])
    def _gen_garbled_table(self):
        """Create the garbled table of a 2-input gate.

        Args:
            operator: The logical function of to the 2-input gate type.
        """
        in_a, in_b, out = self.input[0], self.input[1], self.output
        
        if (self.gate_type=="AND"):
            w_a_0 = self.garbled_values[in_a][0]
            w_a_1 = self.garbled_values[in_a][1]
            w_b_0 = self.garbled_values[in_b][0]
            w_b_1 = self.garbled_values[in_b][1]
            w_i_0, w_i_1,x_0, y_0 = geom.ANDgate(w_a_0,w_a_1,w_b_0, w_b_1)
            c0=geom.Point(x_0, y_0)
            self.garbled_table = c0
            self.garbled_values[out]=(w_i_0,w_i_1)
        elif (self.gate_type=="NAND"):
            w_a_0 = self.garbled_values[in_a][0]
            w_a_1 = self.garbled_values[in_a][1]
            w_b_0 = self.garbled_values[in_b][0]
            w_b_1 = self.garbled_values[in_b][1]
            w_i_0, w_i_1,x_0, y_0 = geom.NANDgate(w_a_0,w_a_1,w_b_0, w_b_1)
            c0=geom.Point(x_0, y_0)
            self.garbled_table = c0
            self.garbled_values[out]=(w_i_0,w_i_1)
        elif (self.gate_type=="OR"):
            w_a_0 = self.garbled_values[in_a][0]
            w_a_1 = self.garbled_values[in_a][1]
            w_b_0 = self.garbled_values[in_b][0]
            w_b_1 = self.garbled_values[in_b][1]
            w_i_0, w_i_1,x_0, y_0 = geom.ORgate(w_a_0,w_a_1,w_b_0, w_b_1)
            c0=geom.Point(x_0, y_0)
            self.garbled_table = c0
            self.garbled_values[out]=(w_i_0,w_i_1)
            
        elif (self.gate_type=="NOR"):
            w_a_0 = self.garbled_values[in_a][0]
            w_a_1 = self.garbled_values[in_a][1]
            w_b_0 = self.garbled_values[in_b][0]
            w_b_1 = self.garbled_values[in_b][1]
            w_i_0, w_i_1,x_0, y_0 = geom.NORgate(w_a_0,w_a_1,w_b_0, w_b_1)
            c0=geom.Point(x_0, y_0)
            self.garbled_table = c0
            self.garbled_values[out]=(w_i_0,w_i_1)
            
        elif (self.gate_type=="XOR"):
            w_a_0 = self.garbled_values[in_a][0]
            w_a_1 = self.garbled_values[in_a][1]
            w_b_0 = self.garbled_values[in_b][0]
            w_b_1 = self.garbled_values[in_b][1]
            w_i_0, w_i_1,x_0, y_0 = geom.XORgate(w_a_0,w_a_1,w_b_0, w_b_1)
            c0=geom.Point(x_0, y_0)
            self.garbled_table = c0
            self.garbled_values[out]=(w_i_0,w_i_1)
        elif (self.gate_type=="XNOR"):
            w_a_0 = self.garbled_values[in_a][0]
            w_a_1 = self.garbled_values[in_a][1]
            w_b_0 = self.garbled_values[in_b][0]
            w_b_1 = self.garbled_values[in_b][1]
            w_i_0, w_i_1,x_0, y_0 = geom.XNORgate(w_a_0,w_a_1,w_b_0, w_b_1)
            c0=geom.Point(x_0, y_0)
            self.garbled_table = c0
            self.garbled_values[out]=(w_i_0,w_i_1)

    def print_garbled_table(self):
        """Print a clear representation of the garbled table."""
        print(f"GATE: {self.output}, TYPE: {self.gate_type}")
        for k, v in self.clear_garbled_table.items():
            # If it's a 2-input gate
            if len(k) > 1:
                garbled_value_a, garbled_value_b, garbled_value_out = v[0], v[1], v[2]
                encr_bit_out = v[3]
                print(f"[{k[0]}, {k[1]}]: "
                      f"[{garbled_value_a[0]}, {garbled_value_a[1]}][{garbled_value_b[0]}, {garbled_value_b[1]}]"
                      f"([{garbled_value_out[0]}, {garbled_value_out[1]}], {encr_bit_out})")
            # Else it's a NOT gate
            else:
                garbled_value_in, garbled_value_out = v[0], v[1]
                encr_bit_out = v[2]
                print(f"[{k[0]}]: "
                      f"[{garbled_value_in[0]}, {garbled_value_in[1]}]"
                      f"([{garbled_value_out[0]}, {garbled_value_out[1]}], {encr_bit_out})")

    def get_garbled_table(self):
        """Return the garbled table of the gate."""
        return self.garbled_table


class GarbledCircuit:
    """A representation of a garbled circuit.

    Args:
        circuit: A dict containing circuit spec.
    """
    def __init__(self, circuit):
        self.circuit = circuit
        self.gates = circuit["gates"]  # list of gates
        self.wires = set()  # list of circuit wires
        
        self.input_wire = set()

        self.garbled_values = {}  # dict of garbled_values
        self.garbled_tables = {}  # dict of garbled tables

        # Retrieve all wire IDs from the circuit
        for gate in self.gates:
            self.wires.add(gate["id"])
            self.wires.update(set(gate["in"]))
        self.wires = list(self.wires)
        self.input_wire.update(set(circuit["alice"] ))
        self.input_wire.update(set(circuit["bob"] ))
        self.input_wire = list(self.input_wire)

        self._gen_garbled_values()
        self._gen_garbled_tables()

    def _gen_garbled_values(self):
        """Create pair of garbled_values for each wire."""
        for wire in self.wires:
            garbled_value0 = random.randint(-2**31,2**31-1)
            garbled_value1 = random.randint(-2**31,2**31-1)
            garbled_value0 = pack('i',garbled_value0)
            garbled_value1 = pack('i',garbled_value1)
            self.garbled_values[wire] = (garbled_value0, garbled_value1)

    def _gen_garbled_tables(self):
        """Create the garbled table of each gate."""
        for gate in self.gates:
            garbled_gate = GarbledGate(gate, self.garbled_values)
            self.garbled_tables[gate["id"]] = garbled_gate.get_garbled_table()

    def print_garbled_tables(self):
        """Print p-bits and a clear representation of all garbled tables."""
        print("print_garbled_tables(self)")
        print(f"======== {self.circuit['id']} ========")
        a3print(f"P-BITS: {self.pbits}")
        for gate in self.gates:
            garbled_table = GarbledGate(gate, self.garbled_values)
            garbled_table.print_garbled_table()
        


    def get_garbled_tables(self):
        """Return dict mapping each gate to its garbled table."""
        return self.garbled_tables

    def get_garbled_values(self):
        """Return dict mapping each wire to its pair of garbled_values."""
        return self.garbled_values
