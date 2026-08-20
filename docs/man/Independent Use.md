<h1>Step-by-step</h1>

<h2>1. Prepare Input Data</h2>

<p><strong>Prepare drivers:</strong></p>
<ul>
  <li>CO<sub>2</sub> concentration</li>
  <li>Regional temperature</li>
  <li>Regional precipitation</li>
  <li>Nitrogen fertilizer application rate</li>
</ul>

<p><strong>Offset the following drivers by their preindustrial levels:</strong></p>
<ul>
  <li>CO<sub>2</sub> concentration</li>
  <li>Regional temperature</li>
  <li>Regional precipitation</li>
</ul>

<h2>2. Load Parameters</h2>

<ul>
  <li>Load parameters from <code>core.Par_crop</code>.</li>
</ul>

<h2>3. Run the Emulator</h2>

<p><strong>Load the CROP emulator:</strong></p>

<pre><code>from core.OSCAR_crop import CROP

CROP = CROP()</code></pre>

<h2>4. Analyze Output</h2>

<br>

<p align="center">
  <big><a href="./Coupling%20with%20OSCAR.md">⬅️ Previous page: Coupling with OSCAR</a></big>
</p>

<p align="center">
  <big><a href="../../README.md">🏠 Home</a></big>
</p>

<p align="center">
  <big><a href="./Coupling%20with%20OSCAR.md">Next page: Coupling with OSCAR ➡️</a></big>
</p>