<h1>Step-by-step</h1>

<h2>1. Prepare Input Data</h2>

<p><strong>Prepare the required drivers:</strong></p>

<ul>
  <li><strong>OSCAR:</strong> anthropogenic emissions, land use, and wood harvest</li>
  <li><strong>CROP emulator:</strong> nitrogen fertilizer application rate</li>
</ul>

<h2>2. Load Parameters</h2>

<ul>
  <li>Load parameters from OSCAR.</li>
  <li>Load parameters from <code>core.Par_crop</code>.</li>
</ul>

<h2>3. Run the Emulator</h2>

<p><strong>Load the food crop emulator:</strong></p>

<pre><code>from core.OSCAR_crop import CROP

CROP = CROP()</code></pre>

<h2>4. Analyze Output</h2>

<br>

<p align="center">
  <big><a href="./Independent%20Use.md">⬅️ Previous page: Independent Use</a></big>
</p>

<p align="center">
  <big><a href="../../README.md">🏠 Home</a></big>
</p>

<p align="center">
  <big><a href="./Independent%20Use.md">Next page: Independent Use ➡️</a></big>
</p>