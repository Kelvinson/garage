#!/usr/bin/env bash
coverage run -m nose2 -c setup.cfg tests.test_fail
coverage xml
bash <(curl -s https://codecov.io/bash)
