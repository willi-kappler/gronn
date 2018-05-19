#!/bin/bash

cd ..
mv gronn.tgz gronn_old.tgz
tar cfvz gronn.tgz -T gronn/file_list.txt
