#!/bin/bash
export fileid=1ZyFQMNGK6gd6MN9cNWAMl0McLL6GHthG
export filename=toulouse_road_network.tar.xz

## WGET ##
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt

wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)

tar xf $filename
rm $filename

rm -f confirm.txt cookies.txt