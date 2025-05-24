import subprocess
import sys

packages = [
    "streamlit",
    "pandas",
    "groq",
    "python-dotenv",
    "numpy",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "scipy",
    "langchain",
    "langchain-groq",
    "langchain-community",
    "openpyxl",
    "reportlab",
    "pillow",
    "kaleido"
]

print("🔍 Instalando paquetes uno por uno para identificar el problemático...\n")

failed_packages = []
successful_packages = []

for i, package in enumerate(packages, 1):
    print(f"[{i}/{len(packages)}] Instalando {package}...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ {package} instalado correctamente\n")
        successful_packages.append(package)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando {package}")
        print(f"   Error: {e.stderr[:200]}...\n")
        failed_packages.append(package)

print("\n" + "="*50)
print("RESUMEN:")
print("="*50)
print(f"\n✅ Paquetes instalados exitosamente ({len(successful_packages)}):")
for pkg in successful_packages:
    print(f"   - {pkg}")

if failed_packages:
    print(f"\n❌ Paquetes con errores ({len(failed_packages)}):")
    for pkg in failed_packages:
        print(f"   - {pkg}")
    
    print("\nSOLUCIONES SUGERIDAS:")
    if "matplotlib" in failed_packages:
        print("- matplotlib: pip install matplotlib --no-build-isolation")
    if "scikit-learn" in failed_packages:
        print("- scikit-learn: pip install scikit-learn --no-deps")
    if "scipy" in failed_packages:
        print("- scipy: pip install scipy --no-build-isolation")
else:
    print("\n🎉 ¡Todos los paquetes se instalaron correctamente!")