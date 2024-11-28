
<p align="center">
  <h2 align="center">Digital Salon: An AI and Physics-Driven Tool for 3D Hair Grooming and Simulation</h2>
  <p align="center">
    <a href="https://cs.yale.edu/homes/che/"><strong>Chengan He<sup>'2*</sup></strong></a>
    ·
    <a href="https://cemse.kaust.edu.sa/people/person/jorge-alejandro-amador-herrera"><strong>Jorge Alejandro Amador Herrera<sup>'3*</sup></strong></a>
    ·
    <a href="https://zhouyisjtu.github.io/"><strong>Yi Zhou<sup>1*†</sup></strong></a>
    ·
    <a href="https://zhixinshu.github.io/"><strong>Zhixin Shu<sup>1</sup></strong></a>
    ·
    <a href="https://www.sunxin.name/"><strong>Xin Sun<sup>1</sup></strong></a>
    ·
    <a href="https://is.mpg.de/~yfeng"><strong>Yao Feng<sup>4,5</sup></strong></a>
    ·
    <a href="https://storage.googleapis.com/pirk.io/index.html"><strong>Sören Pirk<sup>6</sup></strong></a>
    ·
    <a href="http://dmichels.de/"><strong>Dominik L. Michels<sup>3</sup></strong></a>
    ·
    <a href="https://mengzephyr.com/"><strong>Meng Zhang<sup>7</sup></strong></a>
    ·
    <a href="https://tuanfeng.github.io/"><strong>Tuanfeng Y. Wang<sup>1</sup></strong></a>
    ·
    <a href="https://graphics.cs.yale.edu/people/holly-rushmeier"><strong>Holly Rushmeier<sup>2</sup></strong></a>
    <br>
    <br>
        <a href='https://digital-salon.github.io/'><img src="https://img.shields.io/badge/Project_Page-DigitalSalon-green" height=22.5 alt='Project Page'></a>
    <br>
    <b><sup>1</sup> Adobe Research &nbsp; | &nbsp; <sup>2</sup> Yale University &nbsp; | &nbsp; <sup>3</sup> KAUST &nbsp; | &nbsp; <sup>4</sup> Max Planck Institute for Intelligent Systems &nbsp; | &nbsp; <sup>5</sup> ETH Zürich &nbsp; | &nbsp; <sup>6</sup> Kiel University &nbsp; | &nbsp; <sup>7</sup> Nanjing University of Science and Technology </b>
    <br>
    * Equal contribution &nbsp; † Corresponding author &nbsp; ' Previous Adobe Research Intern
  </p>
  
  <table align="center">
    <tr>
    <td>
      <img src="misc/teaser.jpg">
    </td>
    </tr>
  </table>
</p>


## Disclaimer

1. In our live demo at SIGGRAPH Asia 2024, we include renderings based on Adobe in-house data, with all visual content subject to copyright regulations. For this code release, we have replaced the in-house data with renderings from USC-HairSalon, a dataset containing less diversity and complexity that may affect the performance of our program.
2. Due to license issues, we could not include our head mesh in this repo. Please download it from [this link](https://www.avatarneo.com/PinscreenGenericHeadModel.obj) (created by [AVATAR NEO](https://avatarneo.com/)), rename it to `head.obj`, and put it under `data/meshes` before compiling our program.

## Installation

See [INSTALL](INSTALL.md).

## Quick Start

We provide a quick walkthrough in [INSTALL](INSTALL.md), covering text-guided hair generation (copilot) and AI rendering. Meanwhile, all the hairstyles can be interactively simulated and groomed. These operations are supported by Digital Salon with the following keybindings (you need to press `S` to start simulation first):

| Operation                           | Keybinding           |
| ----------------------------------- | -------------------- |
| Start simulation                    | `S`                  |
| Reset framework                     | `R`                  |
| Zoom in and zoom out                | `Scroll Wheel`       |
| Change camera position              | `Right Button`       |
| Change light position               | `ALT` + mouse        |
| Drag head with simulated hair       | `Left Button` + `A`  |
| Select triangles                    | `Left Button` + `B`  |
| Deselect triangles                  | `Left Button` + `N`  |
| Generate hair on selected triangles | `G`                  |
| Cut hair                            | `Right Button` + `A` |
| Reset particles' positions          | `D`                  |

Detailed physical parameters can be tuned within the provided GUI. Recordings of all supported operations are available on our [project page](https://digital-salon.github.io/).

> In text-guided hair generation, some special sentences are reserved for wind field generation, including: `"create some wind?"`, `"add some wind?"`, `"blow some wind?"`, `"create strong wind?"`, and `"add strong wind?"`.

## Acknowledgements

- Our default hairstyle (Jewfro) is adopted from [CT2Hair](https://yuefanshen.net/CTHair). We aligned it with our head mesh and resampled it to ~10k strands.

## Citation

If you found this code useful, please consider citing:
```bibtex
@article{digitalsalon,
    title={Digital Salon: An AI and Physics-Driven Tool for 3D Hair Grooming and Simulation},
    author={He, Chengan and Herrera, Jorge Alejandro Amador and Zhou, Yi and Shu, Zhixin and Sun, Xin and Feng, Yao and Pirk, S{\"o}ren and Michels, Dominik L and Zhang, Meng and Wang, Tuanfeng Y and Rushmeier, Holly},
    url={https://digital-salon.github.io/},
    year={2024}
}
```

## Contact

If you run into any problems or have questions, please create an issue or contact `chengan.he@yale.edu`.