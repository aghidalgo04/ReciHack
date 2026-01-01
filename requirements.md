# Pasos a seguir
### Instalar Grounding DINO
``` bash
git clone https://github.com/IDEA-Research/GroundingDINO
cd GroundingDINO
pip install -e .
cd ..
```

### Conseguir pesos
``` bash
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### Crear carpetas
``` bash
mkdir DINO
cd DINO
mkdir annotations
```

Ejecutar pero cambiar los parametros para que se ejecute en las rutas que quieras
