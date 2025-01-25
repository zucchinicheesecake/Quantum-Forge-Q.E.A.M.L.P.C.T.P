import { useQiskit } from 'react-qiskit';

const QuantumOrderBook = ({ depth }) => {
  const { result } = useQiskit({
    circuit: createOrderBookCircuit(depth),
    backend: 'ibmq_qasm_simulator'
  });

  return (
    <div className="quantum-depth-chart">
      {result && result.map((prob, level) => (
        <DepthBar 
          key={level} 
          probability={prob} 
          level={level} 
        />
      ))}
    </div>
  );
};

export default QuantumOrderBook;
