// Display names follow the existing observation and action order. Values are
// normalized and relative to the inspected model, not screen coordinates.
export const observationLabels = [
  'Ball along court', 'Ball across court', 'Toward/away velocity',
  'Left/right velocity', 'Own paddle position', 'Opponent paddle position',
] as const

export const observationDescriptions = [
  'Centered at 0. Negative toward this model’s end; positive toward its opponent’s end.',
  'Centered at 0. Negative is left; positive is right.',
  'Negative means approaching this model’s paddle; positive means departing. A serve approaching at 5 court units per tick gives −5/6 (about −0.83).',
  'Negative means moving left; positive means moving right. Moving 3 court units right per tick gives +0.5.',
  'This model’s paddle position across the court, on the same centered scale as the ball.',
  'The other paddle’s position across the court, on the same centered scale as the ball.',
] as const

export const actionLabels = ['Left', 'Stay', 'Right'] as const

export const observationValueHelp = 'Position scales run from −1 to +1: own end to opponent’s along the court, left to right across it. Velocity inputs are court units per tick ÷ 6: negative means toward the model’s paddle or left; positive means away or right.'
