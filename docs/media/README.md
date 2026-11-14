# Project images

Install the rendering dependencies from the repository root:

```sh
npm ci
npx playwright install chromium
```

## README screenshot

```sh
npm run docs:screenshot
```

The command builds the app, opens its own local preview and captures the default
Quill–Counter match with live Brain activations. It fixes the match seed and clock
and uses a 1440 × 1000 viewport at device scale 1, capturing the whole page.
It closes its browser and server afterward without changing an open app session.

Review [pongarena.png](pongarena.png) after UI changes, especially navigation,
player labels, controls and input labels.

## Social card

Edit [scripts/social-card.html](../../scripts/social-card.html), then render
[public/social-card.png](../../public/social-card.png):

```sh
npm run docs:social-card
```

The command opens the HTML in its own Chromium browser, waits for the bundled
Manrope font, and captures a 1200 × 630 image at device scale 1.

Keep the tagline **Small brains. Big rallies.** and the subtitle **Play Pong.
Explore neural networks.** Use the app's cream background, forest-green text,
[logo](../../public/mark.svg), vertical court, mint paddle and muted orange
opponent. The network shows Counter's actual `6 → 6 → 4 → 3 → 3` architecture
and 97 parameters, with a thin grid, connections and minimal glow.
