export function QuickRules() {
  return <article className="panel quick-rules">
    <section>
      <span className="section-code">THE MATCH</span>
      <h2>Play to three. Watch five rallies.</h2>
      <p>A miss gives the other player one point. Human games end as soon as someone scores three; drawn rallies add no points and lead to another serve. Human rallies have a 1,600-tick time limit (about 53 seconds). When watching models, the higher score after five rallies wins. Each rally can last 90 seconds, or draw earlier if the models repeat the same state. Draws pause to explain the result before the next serve.</p>
      <p>New games use fresh serves. Restart repeats the same serves for practice. Hit with the edge of your paddle to send a steeper return. The ball speeds up after every return, up to a limit.</p>
    </section>
    <section>
      <span className="section-code">CONTROLS</span>
      <h2>You play from the bottom.</h2>
      <p><strong>Keyboard:</strong> A/D or ←/→ to move. P to play or pause, R to restart, F for full screen. Esc exits full screen or releases mouse control.</p>
      <p><strong>Mouse or trackpad:</strong> click the court, then move left and right to steer. Click outside it to release control.</p>
      <p><strong>Touch:</strong> hold the left or right half of the court to move; lift to stop. Every control uses the same paddle speed as the models.</p>
    </section>
    <section>
      <span className="section-code">INSIDE THE GAME</span>
      <h2>Six observations. Three moves.</h2>
      <p>Every model sees the ball’s position and velocity, its own paddle, and its opponent’s paddle. Thirty times a second, it chooses left, stay, or right. Pause a match to inspect the real neuron activations in its brain panel.</p>
      <p>The League keeps five rallies per game and a 1,600-tick rally limit, playing each matchup from both sides. It awards one point for winning more rallies across the two games, or half a point for a tie.</p>
      <div className="finding-actions">
        <a className="secondary-btn" href="https://github.com/jonaslindblom/pongarena/blob/main/docs/RULES.md">Full physics & model contract</a>
        <a className="secondary-btn" href="https://github.com/jonaslindblom/pongarena/blob/main/CONTRIBUTING.md">Submit your own model</a>
      </div>
    </section>
  </article>
}
