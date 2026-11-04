import type { Controller } from './models'

export function architectureLabel(model: Controller | undefined) {
  const graph = model?.architecture
  return graph?.widths?.join(' → ') ?? (graph ? `${graph.operations.length} operator graph` : '')
}

export function Architecture({ model }: { model: Controller }) {
  const graph = model.architecture
  if (!graph) return null
  const activations = [
    ...new Set(graph.operations.map((n) => n.op).filter((op) => ['Relu', 'Tanh'].includes(op))),
  ]
  return (
    <div className="architecture">
      <div className="architecture-caption">NETWORK ARCHITECTURE</div>
      <div
        className="architecture-chain"
        aria-label={'Network dimensions ' + architectureLabel(model)}
      >
        {graph.widths ? (
          graph.widths.map((width, i) => (
            <span key={i} className="architecture-layer">
              {i > 0 && <i aria-hidden="true">→</i>}
              <b>
                {width}
                <small>
                  {i === 0 ? 'INPUT' : i === graph.widths!.length - 1 ? 'OUTPUT' : 'HIDDEN'}
                </small>
              </b>
            </span>
          ))
        ) : (
          <b>{architectureLabel(model)}</b>
        )}
      </div>
      <div className="architecture-activation">
        {activations.join(' · ').replace('Relu', 'ReLU') || 'Linear'} · float32
      </div>
      <details className="architecture-details">
        <summary>Inspect ONNX graph</summary>
        <ol>
          {graph.operations.map((node) => (
            <li key={node.output}>
              <b>
                {node.op} <span>[{node.shape.join(' × ')}]</span>
              </b>
              <code>
                {node.inputs.join(', ')} → {node.output}
              </code>
            </li>
          ))}
        </ol>
      </details>
    </div>
  )
}
