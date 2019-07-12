#!/bin/bash
set -o errexit
base_path=$(cd `dirname $0`; pwd)
cd $base_path

shopt -s expand_aliases
alias hub="python $base_path/../paddlehub/commands/hub.py "$@""

# test install command
hub install lac

# test show command
hub show lac

# test list command
hub list

# test version command
hub version

# test help command
hub help

# test uninstall command
hub uninstall lac

# test clear command
hub clear

# test search command
hub search lac

# test run command
hub run lac --input_text "今天天气不错"

# test download command
hub download lac
