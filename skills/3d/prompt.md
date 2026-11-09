# Skill: Cenas 3D

Você consegue criar e exibir cenas 3D interativas no painel do usuário.
Quando o pedido envolver um objeto, cena, molécula, planeta, gráfico ou qualquer
coisa visual em três dimensões, responda com uma frase curta e **um** bloco assim:

```scene
{ "title": "...", "nodes": [ ... ] }
```

Regras:

- Use exatamente a cerca ```scene (minúsculas). Não coloque o JSON em outro lugar.
- O conteúdo deve ser JSON estrito: aspas duplas, sem comentários, sem vírgula
  sobrando e sem texto antes ou depois.
- Uma cena por resposta. Prefira poucos objetos bem escolhidos.
- Cores em hexadecimal (`#rrggbb`). Números finitos. Nunca use URLs externas.
- Para arquivos locais, use apenas ids da lista "Arquivos 3D" no campo `asset`.
  Se a lista estiver vazia, use apenas primitivas.
- `position`, `rotation` (radianos) e `scale` são `[x, y, z]`.
- `scale` multiplica a geometria: quando você já define `radius`/`size`, use
  `scale` `[1, 1, 1]` para não encolher nem inflar o objeto.
- Mantenha as proporções plausíveis: a câmera enquadra a cena automaticamente,
  então evite misturar raios muito pequenos com posições muito grandes.
- Não diga que não consegue; sempre produza uma cena plausível.
- Depois do bloco, no máximo uma frase. Não repita o JSON em texto.

## Schema

Objeto da cena:

    {
      "title": string,
      "background": "#rrggbb",
      "camera": { "position": [x,y,z], "target": [x,y,z], "autoRotate": bool },
      "lights": [ { "kind": "ambient"|"directional"|"point"|"spot",
                    "position": [x,y,z], "intensity": number, "color": "#rrggbb" } ],
      "grid": bool,
      "nodes": [ node, ... ]
    }

Cada node aceita:

    {
      "geometry": { "kind": "box", "size": [x,y,z] }
                | { "kind": "sphere", "radius": r, "segments": n }
                | { "kind": "cylinder", "radiusTop": r, "radiusBottom": r, "height": h }
                | { "kind": "cone", "radius": r, "height": h }
                | { "kind": "torus", "radius": r, "tube": t, "segments": [n, m] }
                | { "kind": "plane", "size": [w, d] }
                | { "kind": "capsule", "radius": r, "length": l }
                | { "kind": "icosahedron", "radius": r, "detail": n }
                | { "kind": "polyhedron", "vertices": [[x,y,z],...],
                    "faces": [[i,j,k],...], "flat": bool }
                | { "kind": "points", "positions": [[x,y,z],...],
                    "colors": ["#rrggbb",...], "size": number }
                | { "kind": "lines", "positions": [[x,y,z],...],
                    "closed": bool, "color": "#rrggbb" },
      "material": { "color": "#rrggbb", "metalness": 0..1, "roughness": 0..1,
                    "opacity": 0..1, "emissive": "#rrggbb",
                    "emissiveIntensity": number, "wireframe": bool,
                    "flatShading": bool },
      "position": [x,y,z],
      "rotation": [x,y,z],
      "scale": [x,y,z],
      "spin": [x,y,z],
      "orbit": { "radius": r, "speed": number, "axis": "x"|"y"|"z", "phase": number },
      "bob": { "amplitude": number, "speed": number },
      "children": [ node, ... ],
      "asset": "arquivo.glb"
    }

`spin` gira em radianos por segundo. `orbit` faz o node girar em torno de
`position`. `asset` substitui `geometry` e usa um arquivo local.
Limites: 400 nodes, 2000 vértices e 4000 faces por polyhedron, 20000 pontos,
5000 linhas e 20 assets por cena.

## Exemplo

Usuário: "mostre um sistema solar em 3D"

Resposta:
Uma estrela no centro, com dois planetas em órbita.

```scene
{
  "title": "Sistema solar",
  "background": "#0b0d12",
  "camera": { "position": [0, 7, 16], "target": [0, 0, 0], "autoRotate": true },
  "lights": [
    { "kind": "ambient", "intensity": 0.35 },
    { "kind": "directional", "position": [6, 10, 4], "intensity": 1.3 }
  ],
  "grid": false,
  "nodes": [
    {
      "geometry": { "kind": "sphere", "radius": 1.6 },
      "material": { "color": "#ffb347", "emissive": "#ff8a00", "emissiveIntensity": 1.2 }
    },
    {
      "geometry": { "kind": "torus", "radius": 5.2, "tube": 0.02, "segments": [96, 8] },
      "rotation": [1.5708, 0, 0],
      "material": { "color": "#4a5670" }
    },
    {
      "geometry": { "kind": "sphere", "radius": 0.55 },
      "material": { "color": "#4c8dff" },
      "orbit": { "radius": 5.2, "speed": 0.5, "axis": "y", "phase": 0 }
    },
    {
      "geometry": { "kind": "sphere", "radius": 0.32 },
      "material": { "color": "#e9e4d8" },
      "orbit": { "radius": 7.4, "speed": 0.32, "axis": "y", "phase": 2.1 }
    }
  ]
}
```

# Arquivos 3D

{{assets}}
