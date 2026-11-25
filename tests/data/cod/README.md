# Real-world CIF test fixtures from COD

These CIF files were downloaded from the [Crystallography Open Database (COD)](https://www.crystallography.net/cod/) to test `crystal-mcp` against real experimental structures containing partial occupancies, mixed sites, split sites, numerical uncertainties, hydrogen-bearing molecular crystals, and triclinic cells.

| COD ID | Chemical Formula | Space Group | Conditions Covered | Source URL |
| :--- | :--- | :--- | :--- | :--- |
| `1000030` | `Ba2 Cu3 O6.9 Y` (YBa2Cu3O6.9) | `P m m m` (No. 47) | 部分占有サイト（`_atom_site_occupancy = 0.910(8)`）、不確かさ付き数値（格子定数・座標・占有率） | https://www.crystallography.net/cod/1000030.html |
| `1000068` | `Li Nb O6 W` | `P -4 21 m` (No. 113) | 不規則な混合サイト（同一座標に Nb 0.892 / W 0.108 および W 0.892 / Nb 0.108）、不確かさ付き数値 | https://www.crystallography.net/cod/1000068.html |
| `1530416` | `H2 K O4 P` (KDP) | `I -4 2 d` (No. 122) | 分割サイト（0.358 Å 離れた2位置に H が占有率 0.5 ずつ）、部分占有サイト | https://www.crystallography.net/cod/1530416.html |
| `1008785` | `C H4 N2 O` (Urea) | `P -4 21 m` (No. 113) | 水素を含む分子性結晶（中性子回折）、不確かさ付き数値 | https://www.crystallography.net/cod/1008785.html |
| `1100710` | `C8 H16 Au2 N8 Pt` | `P -1` (No. 2) | 三斜晶（`P-1`）、水素を含む分子性／錯体結晶（SHELXL-97 出力）、不確かさ付き数値 | https://www.crystallography.net/cod/1100710.html |
| `9001735` | `As2 Bi2 Cu0.88 Fe1.12 O12` (Medenbachite) | `P -1` (No. 2) | 三斜晶（`P-1`）、分割サイト（0.463 Å 離れた Bi1/Bi2 に占有率 0.5 ずつ）、不規則な混合サイト（Cu 0.88 / Fe 0.12） | https://www.crystallography.net/cod/9001735.html |

## Provenance / License

- All CIF files are obtained from COD (`https://www.crystallography.net/cod/<ID>.cif`, public domain).
- `9001735.cif` originates from AMCSD (`0001791`, Krause et al., *American Mineralogist* 81, 505–512, 1996) via COD.
