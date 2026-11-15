export const presets = [
  {
    id: 'quill', name: 'Quill', kind: 'Programmed weights', handcrafted: true, submittedBy: 'jonaslindblom',
    description: 'A 163-parameter network whose dense ReLU layers predict the ball’s landing point and choose a paddle position. Codex developed its weight-generating formulas and iteratively revised them using simulated match feedback.',
  },
  {
    id: 'small', name: 'Small', kind: 'Trained weights', submittedBy: 'jonaslindblom',
    description: 'A trained 163-parameter network with exactly Quill’s layer layout. Distillation from Large, reinforcement learning and evolutionary search produced its weights; Quill and Counter were held out of training and checkpoint selection.',
  },
  {
    id: 'large', name: 'Large', kind: 'Trained weights', submittedBy: 'jonaslindblom',
    description: 'A larger neural network trained in PyTorch from a reproducible recipe. It receives the same observations and uses the same movement limits as every other player.',
  },
  {
    id: 'predict', name: 'Predict', kind: 'Programmed weights', handcrafted: true, submittedBy: 'jonaslindblom',
    description: 'A 75-parameter programmed ReLU network that predicts the ball’s landing point and reflects one wall bounce. It uses 63 MACs per decision, without Quill’s recovery gate or opponent-dependent aiming.',
  },
  {
    id: 'attack', name: 'Attack', kind: 'Programmed weights', handcrafted: true, submittedBy: 'jonaslindblom',
    description: 'An aggressive 75-parameter fork of Predict that forecasts a bounded landing point, reflects a wall bounce and aims away from the opponent. Its weight-generating formulas were developed with Codex and refined using simulated match feedback.',
  },
  {
    id: 'counter', name: 'Counter', kind: 'Programmed weights', handcrafted: true, submittedBy: 'jonaslindblom',
    description: 'A 97-parameter programmed network with a longer landing forecast, opponent-aware aiming and center recovery. Its weight-generating formulas were developed with Codex and refined using simulated match feedback.',
  },
  {
    id: 'follow', name: 'Follow', kind: 'Programmed weights', handcrafted: true, submittedBy: 'jonaslindblom',
    description: 'Thirteen parameters and one hidden neuron: Follow moves toward the ball’s current position. Its programmed weights are the first example in How it thinks. It does not predict bounces or aim returns.',
  },
  {
    id: 'oneshot', name: 'OneShot', kind: 'Prompted weights', submittedBy: 'jonaslindblom',
    description: 'Astra returned these 163 weights and biases in one reply to the game rules and a tactical brief.',
  },
]
