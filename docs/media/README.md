# README screenshot

From the repository root:

```sh
npm ci
npx playwright install chromium
npm run docs:screenshot
```

The command builds the app, opens its own local preview and captures the default
Quill–Counter match with live Brain activations. It fixes the match seed and clock
and uses a 1440 × 1000 viewport at device scale 1, capturing the whole page.
It closes its browser and server afterward without changing an open app session.

Review [pongarena.png](pongarena.png) after UI changes, especially navigation,
player labels, controls and input labels. The social card has a separate
[editable source and rendering recipe](../social-card-prompt.txt).
