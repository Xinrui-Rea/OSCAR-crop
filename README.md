<h1>🌾 OSCAR-crop</h1>

<p>A crop emulator calibrated on GGCMI Phase 3 models.</p>

<h2>🏗️ Model Architecture</h2>

<p align="center">
  <img src="./docs/sci/crop_fd/Model%20Architecture.png" alt="Food Crop Emulator Architecture">
</p>

<h2>✨ Features</h2>

<ul>
  <li>⚡ <strong>High efficiency</strong></li>
  <li>🗺️ <strong>Flexible regional aggregation</strong></li>
</ul>

<h2>🚀 Installation</h2>

<p>
  Since this repository is not packaged for <code>pip</code>, clone the repository
  and run it directly within a Python environment.
</p>

<h3>Prerequisites and Dependencies</h3>

<p>
  This project was developed and tested on <strong>Python 3.12</strong>
  and requires <strong>Python 3.11</strong> or higher.
</p>

<p>
  Required packages and version constraints are listed in
  <a href="./requirements.txt"><code>requirements.txt</code></a>.
</p>

<ul>
  <li><strong>Data and Science:</strong> <code>numpy</code>, <code>pandas</code>, <code>scipy</code>, <code>xarray</code></li>
  <li><strong>Visualization:</strong> <code>matplotlib</code>, <code>cartopy</code></li>
</ul>

<h3>Step-by-step Installation</h3>

<h4>1. Create and activate a virtual environment</h4>

<pre><code># Create the environment
python3 -m venv env

# Activate the environment on Linux/macOS
source env/bin/activate</code></pre>

<h4>2. Clone the repository</h4>

<pre><code>git clone https://github.com/Xinrui-Rea/OSCAR-crop.git
cd OSCAR-crop</code></pre>

<h4>3. Install the required packages</h4>

<pre><code>pip install -r requirements.txt</code></pre>

<h2>📄 Documentation</h2>

<ul>
  <li><a href="./docs/man/Independent%20Use.md"><strong>Independent Use</strong></a></li>
  <li><a href="./docs/man/Coupling%20with%20OSCAR.md"><strong>Coupling with OSCAR</strong></a></li>
</ul>

<h2>📚 Scientific Guide</h2>

<h3>Food Crop Emulator</h3>

<ul>
  <li><a href="./docs/sci/crop_fd/Acronyms.md"><strong>Acronyms</strong></a></li>
  <li><a href="./docs/sci/crop_fd/Dimensions.md"><strong>Dimensions</strong></a></li>
  <li><a href="./docs/sci/crop_fd/Parameters.md"><strong>Parameters</strong></a></li>
  <li><a href="./docs/sci/crop_fd/Variables.md"><strong>Variables</strong></a></li>
</ul>

<h2>📝 Notes</h2>

<p>
  The current model structure is consistent with OSCARv3.
  The regional aggregation scheme has been updated to OSCARv4.
</p>

<h2>📖 Citation</h2>

<p>If you use this code or data in your research, please cite:</p>

<p>
  <a href="https://doi.org/10.5194/gmd-19-5857-2026">
    Liu, X. et al. A food crop yield emulator for integration in the compact Earth
    system model OSCAR (OSCAR-crop v1.0). Geosci Model Dev 19, 5857–5880 (2026).
  </a>
</p>

<h2>📞 Contact</h2>

<ul>
  <li><strong>Developer:</strong> Xinrui Liu</li>
  <li><strong>Email:</strong> <a href="mailto:liuxinrui@iiasa.ac.at">liuxinrui@iiasa.ac.at</a></li>
</ul>