SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python3 "$SCRIPT_DIR/update_version_in_cargo_toml.py" 0.0.0
python3 "$SCRIPT_DIR/update_version_in_cargo_toml.py"
cd "$SCRIPT_DIR/.." || exit
maturin develop --release --uv --target "$(rustc -vV | sed -n 's|host: ||p')"
python3 "$SCRIPT_DIR/update_version_in_cargo_toml.py" 0.0.0
