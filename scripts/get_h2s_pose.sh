#!/bin/bash
# Download the How2Sign pose_format zip parts from the Uni-Sign HF repo, join, list contents.
set -e
D=$(cd "$(dirname "$0")/.." && pwd)/dataset/How2Sign_dl
mkdir -p $D && cd $D
for p in 00 01 02 03; do
    [ -f how2sign_pose_format.zip.$p ] || curl -sSL --retry 3 -o how2sign_pose_format.zip.$p.part \
        "https://huggingface.co/ZechengLi19/Uni-Sign/resolve/main/how2sign_pose_format.zip.$p" \
        && mv how2sign_pose_format.zip.$p.part how2sign_pose_format.zip.$p
    echo "part $p $(stat -c %s how2sign_pose_format.zip.$p) $(date)"
done
cat how2sign_pose_format.zip.0? > how2sign_pose_format.zip && rm how2sign_pose_format.zip.0?
python3 -c "import zipfile; z=zipfile.ZipFile('how2sign_pose_format.zip'); n=z.namelist(); print('entries', len(n)); print(n[:5]); print(n[-3:])"
echo "downloaded $(date)"
