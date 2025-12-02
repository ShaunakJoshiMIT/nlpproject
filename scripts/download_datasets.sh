#!/bin/bash
# Downloads the datasets

BASEDIR=$(dirname "$0")
BASEDIR=$(dirname "$BASEDIR")
cd "$BASEDIR" || exit
mkdir -p data
cd data || exit

# MAESTRO (v3.0.0) (only MIDI)
curl -LJO https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip || wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip maestro-v3.0.0-midi.zip && rm maestro-v3.0.0-midi.zip
mv maestro-v3.0.0 Maestro
# shellcheck disable=SC2044
for file in $(find "Maestro" -type f -name "*.midi"); do
    mv -- "$file" "${file%.midi}.mid"
done
python ../scripts/preprocess_maestro.py

# MMD (commented out - not using MMD dataset)
# python ../scripts/clean_mmd.py

# PREPROCESS FOR OCTUPLE
python ../scripts/preprocess_for_octuple.py