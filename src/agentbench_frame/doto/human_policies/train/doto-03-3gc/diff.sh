#!/bin/bash
set -v on
diff Attack.cpp ../../../AI/Attack.cpp -s -b
diff Map.cpp ../../../AI/Map.cpp -s -b
diff TaskScheduler.cpp ../../../AI/TaskScheduler.cpp -s -b
