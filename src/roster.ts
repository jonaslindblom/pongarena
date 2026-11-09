export const presets = [
  {
    id: 'quill', name: 'Quill', kind: 'Handwritten weights', handcrafted: true, submittedBy: 'jonaslindblom',
    description: 'A 163-parameter neural network written by hand. Its dense ReLU layers predict the ball’s landing point and choose a paddle position. No optimizer or training data is used to construct its weights.',
  },
  {
    id: 'small', name: 'Small', kind: 'Trained weights', submittedBy: 'jonaslindblom',
    description: 'A compact neural network trained in PyTorch from a reproducible recipe, within the 256-parameter Small class.',
  },
  {
    id: 'large', name: 'Large', kind: 'Trained weights', submittedBy: 'jonaslindblom',
    description: 'A larger neural network trained in PyTorch from a reproducible recipe. It receives the same observations and uses the same movement limits as every other player.',
  },
  {
    id: 'predict', name: 'Predict', kind: 'Handwritten weights', handcrafted: true, submittedBy: 'jonaslindblom',
    description: 'A 75-parameter handwritten ReLU network that predicts the ball’s landing point and reflects one wall bounce. It uses 63 MACs per decision, without Quill’s recovery gate or opponent-dependent aiming.',
  },
  {
    id: 'attack', name: 'Attack', kind: 'Handwritten weights', handcrafted: true, submittedBy: 'jonaslindblom',
    description: 'An aggressive 75-parameter fork of Predict. Its handwritten ReLU weights predict a bounded landing point, reflect a wall bounce and offset contact to aim toward a corner away from the opponent. No training or automated fitting.',
  },
  {
    id: 'counter', name: 'Counter', kind: 'Handwritten weights', handcrafted: true, submittedBy: 'jonaslindblom',
    description: 'A 97-parameter handwritten network with a longer landing forecast, opponent-aware corner aiming and center recovery while the ball travels away. Its four dense layers use no training or automated fitting.',
  },
  {
    id: 'follow', name: 'Follow', kind: 'Handwritten weights', handcrafted: true, submittedBy: 'jonaslindblom',
    description: 'Thirteen parameters and one hidden neuron: Follow moves toward the ball’s current position. Its handwritten weights are the first example in How it thinks. It does not predict bounces or aim returns.',
  },
]
