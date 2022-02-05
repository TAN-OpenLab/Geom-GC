#!/usr/bin/env python3
import logging
import ot
import util
import GC
from abc import ABC, abstractmethod
import geom


logging.basicConfig(format="[%(levelname)s] %(message)s",
                    level=logging.WARNING)


class Generator(ABC):
    """An abstract class for Generators (e.g. Alice)."""
    def __init__(self, circuits):
        
        circuits = util.parse_json(circuits)
        self.name = circuits["name"]
        self.circuits = []

        for circuit in circuits["circuits"]:
            garbled_circuit = GC.GarbledCircuit(circuit)

            entry = {
                "circuit": circuit,
                "garbled_circuit": garbled_circuit,
                "garbled_tables": garbled_circuit.get_garbled_tables(),
                "garbled_values": garbled_circuit.get_garbled_values(),
           
            }

            self.circuits.append(entry)  

    @abstractmethod
    def start(self):
        pass


class Alice(Generator):
    """Alice is the creator of the Geom-GC.

    Alice creates a garbled circuit and sends it to the evaluator along with her
    input garbled values. Alice will finally print the truth table of the circuit
    for all combination of Alice-Bob inputs.

    Alice does not know Bob's inputs but for the purpose
    of printing the truth table only, Alice assumes that Bob's inputs follow
    a specific order.

    Attributes:
        circuits: the JSON file containing circuits
        oblivious_transfer: Optional; enable the Oblivious Transfer protocol
            (True by default).
    
    """
    def __init__(self, circuits, oblivious_transfer=True):
        super().__init__(circuits)
        self.socket = util.GeneratorSocket()
        self.ot = ot.ObliviousTransfer(self.socket, enabled=oblivious_transfer)

    def start(self):
        """Start Geom protocol."""
        for circuit in self.circuits:
            to_send = {
                "circuit": circuit["circuit"],
                "garbled_tables": circuit["garbled_tables"],
            }
            logging.debug(f"Sending {circuit['circuit']['id']}")
            self.socket.send_wait(to_send)   
            self.print(circuit)

    def print(self, entry):
        """Print circuit evaluation for all Bob and Alice inputs.

        Args:
            entry: A dict representing the circuit to evaluate.
        """
        circuit, garbled_values = entry["circuit"], entry["garbled_values"]
        outputs = circuit["out"]
        a_wires = circuit.get("alice", [])  # Alice's wires
        a_inputs = {}  # map from Alice's wires to garbled inputs
        b_wires = circuit.get("bob", [])  # Bob's wires
        
        b_garbled_values = {  # map from Bob's wires to a pair garbled values
            w: (garbled_value0, garbled_value1)
            for w, (garbled_value0, garbled_value1) in garbled_values.items() if w in b_wires
        }

        result2 = {}   

        N = len(a_wires) + len(b_wires)

        print(f"======== {circuit['id']} ========")

        # Generate all inputs for both Alice and Bob
        for bits in [format(n, 'b').zfill(N) for n in range(2**N)]:
            bits_a = [int(b) for b in bits[:len(a_wires)]]  # Alice's inputs

            # Map Alice's wires to (garbled_value, encr_bit)
            for i in range(len(a_wires)):
                a_inputs[a_wires[i]] = (garbled_values[a_wires[i]][bits_a[i]])

            # Send Alice's encrypted inputs and garbled_values to Bob
            result = self.ot.get_result(a_inputs, b_garbled_values)  #garbled outputs
            for w in outputs:
                if result[w]==garbled_values[w][0]:
                    result2[w]=0
                elif result[w]==garbled_values[w][1]:
                    result2[w]=1
                else:
                    print ("decoding error!")

            # Format output
            str_bits_a = ' '.join(bits[:len(a_wires)])
            str_bits_b = ' '.join(bits[len(a_wires):])
            str_result = ' '.join([str(result2[w]) for w in outputs])

            print(f"  Alice{a_wires} = {str_bits_a} "
                  f"Bob{b_wires} = {str_bits_b}  "
                  f"Outputs{outputs} = {str_result}")

        print()


class Bob:
    """Bob is the receiver and evaluator of the circuit.

    Bob receives the garbled circuit from Alice, computes the results and sends
    them back.

    Args:
        oblivious_transfer: Optional; enable the Oblivious Transfer protocol
            (True by default).
    """
    def __init__(self, oblivious_transfer=True):
        self.socket = util.EvaluatorSocket()
        self.ot = ot.ObliviousTransfer(self.socket, enabled=oblivious_transfer)
    def listen(self):
        """Start listening for Alice messages."""
        logging.info("Start listening")
        print ("Start listening")
        while True:
            try:
                entry = self.socket.receive()  
                self.socket.send(True)  
                
                self.send_evaluation(entry)
            except KeyboardInterrupt:
                logging.info("Stop listening")
                break

    def send_evaluation(self, entry):
        """Evaluate circuit for all Bob and Alice's inputs and
        send back the results.

        Args:
            entry: A dict representing the circuit to evaluate.
        """
        circuit = entry["circuit"]
        garbled_tables = entry["garbled_tables"]
        a_wires = circuit.get("alice", [])  # list of Alice's wires
        b_wires = circuit.get("bob", [])  # list of Bob's wires
        N = len(a_wires) + len(b_wires)

        print(f"Received {circuit['id']}")

        # Generate all possible inputs for both Alice and Bob
        for bits in [format(n, 'b').zfill(N) for n in range(2**N)]:
            bits_b = [int(b) for b in bits[N - len(b_wires):]]  # Bob's inputs

            # Create dict mapping each wire of Bob to Bob's input
            b_inputs_clear = {
                b_wires[i]: bits_b[i]
                for i in range(len(b_wires))
            }
            # Evaluate and send result to Alice
            self.ot.send_result(circuit, garbled_tables,
                                b_inputs_clear)


class LocalTest(Generator):
    """A class for local tests.

    Print a circuit evaluation or garbled tables.

    Args:
        circuits: the JSON file containing circuits
        print_mode: Print a clear version of the garbled tables or
            the circuit evaluation (the default).
    """
    def __init__(self, circuits, print_mode="circuit"):
        super().__init__(circuits)
        self._print_mode = print_mode
        self.modes = {
            "circuit": self._print_evaluation,
            "table": self._print_tables,
        }
        logging.info(f"Print mode: {print_mode}")

    def start(self):
        """Start local protocol."""
        for circuit in self.circuits:
            self.modes[self.print_mode](circuit)

    def _print_tables(self, entry):
        """Print garbled tables."""
        entry["garbled_circuit"].print_garbled_tables()

    def _print_evaluation(self, entry):
        """Print circuit evaluation."""

        circuit, garbled_values = entry["circuit"], entry["garbled_values"]
        garbled_tables = entry["garbled_tables"]
        outputs = circuit["out"]
        a_wires = circuit.get("alice", [])  # Alice's wires
        a_inputs = {}  # map from Alice's wires to garbled inputs
        b_wires = circuit.get("bob", [])  # Bob's wires
        b_inputs = {}  # map from Bob's wires to garbled inputs
  
        result2 = {}    
  
  
        N = len(a_wires) + len(b_wires)

        print(f"======== {circuit['id']} ========")

        # Generate all possible inputs for both Alice and Bob
        for bits in [format(n, 'b').zfill(N) for n in range(2**N)]:
            bits_a = [int(b) for b in bits[:len(a_wires)]]  # Alice's inputs
            bits_b = [int(b) for b in bits[N - len(b_wires):]]  # Bob's inputs


            for i in range(len(a_wires)):
                a_inputs[a_wires[i]] = garbled_values[a_wires[i]][bits_a[i]]
                                        


            for i in range(len(b_wires)):
                b_inputs[b_wires[i]] = garbled_values[b_wires[i]][bits_b[i]]

            result = GC.evaluate(circuit, garbled_tables, a_inputs,
                                  b_inputs)
            for w in outputs:
                if result[w]==garbled_values[w][0]:
                    result2[w]=0
                elif result[w]==garbled_values[w][1]:
                    result2[w]=1
                else:
                    print ("decoding error!")
            # Format output
            str_bits_a = ' '.join(bits[:len(a_wires)])
            str_bits_b = ' '.join(bits[len(a_wires):])
            str_result = ' '.join([str(result2[w]) for w in outputs])

            print(f"  Alice{a_wires} = {str_bits_a} "
                  f"Bob{b_wires} = {str_bits_b}  "
                  f"Outputs{outputs} = {str_result}")

        print()

    @property
    def print_mode(self):
        return self._print_mode

    @print_mode.setter
    def print_mode(self, print_mode):
        if print_mode not in self.modes:
            logging.error(f"Unknown print mode '{print_mode}', "
                          f"must be in {list(self.modes.keys())}")
            return
        self._print_mode = print_mode


def main(
    party,
    circuit_path="circuits/default.json",
    oblivious_transfer=True,
    print_mode="circuit",
    loglevel=logging.WARNING,
):
    logging.getLogger().setLevel(loglevel)

    if party == "alice":
        alice = Alice(circuit_path, oblivious_transfer=oblivious_transfer)
        print ("circuit_path, oblivious_transfer", circuit_path, oblivious_transfer)
        alice.start()
    elif party == "bob":
        bob = Bob(oblivious_transfer=oblivious_transfer)
        bob.listen()
    elif party == "local":
        local = LocalTest(circuit_path, print_mode=print_mode)
        local.start()
    else:
        logging.error(f"Unknown party '{party}'")


if __name__ == '__main__':
    import argparse

    def init():
        loglevels = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }

        parser = argparse.ArgumentParser(description="Run Geom-GC protocol.")
        parser.add_argument("party",
                            choices=["alice", "bob", "local"],
                            help="the party to run")
        parser.add_argument(
            "-c",
            "--circuit",
            metavar="circuit.json",
            default="circuits/default.json",
            help=("the JSON circuit file for alice and local tests"),
        )
        parser.add_argument("--no-oblivious-transfer",
                            action="store_true",
                            help="disable oblivious transfer")
        parser.add_argument(
            "-m",
            metavar="mode",
            choices=["circuit", "table"],
            default="circuit",
            help="the print mode for local tests (default 'circuit')")
        parser.add_argument("-l",
                            "--loglevel",
                            metavar="level",
                            choices=loglevels.keys(),
                            default="warning",
                            help="the log level (default 'warning')")

        main(
            party=parser.parse_args().party,
            circuit_path=parser.parse_args().circuit,
            oblivious_transfer=not parser.parse_args().no_oblivious_transfer,
            print_mode=parser.parse_args().m,
            loglevel=loglevels[parser.parse_args().loglevel],
        )
        print (parser.parse_args().party)
        print (parser.parse_args().circuit,parser.parse_args().no_oblivious_transfer)
        print (logging.DEBUG,logging.INFO,logging.WARNING,logging.ERROR)
    init()
    
