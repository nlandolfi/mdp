#!/bin/bash

protoc --proto_path=$GOPATH/src/ \
       $GOPATH/src/github.com/nlandolfi/mdp/mdp.proto --go_out=plugins=grpc:$GOPATH/src/
