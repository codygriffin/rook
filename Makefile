#-------------------------------------------------------------------------------
#
# rook::Makefile
#
#  The MIT License (MIT)
#
#  Copyright (C) 2014 Cody Griffin (cody.m.griffin@gmail.com)
# 
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#

# Stupid OS X - this should probably be an environment var
CC      := /usr/local/bin/g++-4.9 

# Some lovely directories
BIN_DIR := ./bin
OBJ_DIR := ./obj
DEP_DIR := ./dep
DAT_DIR := ./data
LIB_DIR := ./lib
INC_DIR := ./inc
TST_DIR := ./tst
SRC_DIR := ./src

# Directories that are NOT in version control
BUILD_DIRS := $(BIN_DIR) $(OBJ_DIR) $(DEP_DIR) $(DAT_DIR)

# Some lovely CFLAGs
CFLAGS  := -std=gnu++0x

# Some lovely DEBUG options (make DEBUG=1)
ifdef DEBUG
  CFLAGS  += -Og -g -pg 
else
  CFLAGS  += -O3 
endif

# Default - run all tests
default: all

# Clean up binaries
clean:
	rm -rf ./bin/*
	rm -f  ./dep/*
	rm -f  ./lib/*
	rm -f  ./obj/*

all::
	# Running all tests...
.PHONY: all

# Include our dependency targets
-include $(DEP_DIR)/*.d

# Retrieve MNIST data from Yann's site
MNIST_URL  := http://yann.lecun.com/exdb/mnist
MNIST_DATA := t10k-images-idx3-ubyte  \
              t10k-labels-idx1-ubyte  \
              train-images-idx3-ubyte \
              train-labels-idx1-ubyte
mnist::
	# Getting mnist data...
.PHONY: mnist

#-------------------------------------------------------------------------------
#
# TEST_CAST(name, source, negative-cases, other dependencies)
# 
# Really simple MACRO for defining test-cases.  More specifically, compile-time
# test cases (runtime tests can just exit with a non-zero value on failure)
# 
define TEST_CASE

# Building $1 tests
$(BIN_DIR)/$(1): $2 $4 | $(BIN_DIR) $(DEP_DIR) $(OBJ_DIR) 
	$(CC) $(CFLAGS) -o $$@ $2 -pthread -L$(LIB_DIR) -I$$(INC_DIR) -MMD -MT $$@ -MF $(DEP_DIR)/$$(notdir $$(basename $$<)).d 

# Test case target 
test/$1: $$(BIN_DIR)/$1
	# Executing run-time tests
	$$<

	# Executing compile-time tests
	if [ $$(words $3) -ne 0 ]; \
	then \
		! $(CC) $(CFLAGS) -pthread $2 -I$$(INC_DIR)/ $$(foreach failcase,$3, -D$$(failcase)) &> /dev/null; \
	fi; 
	# All tests succeeded!

# Add the new test case to our 'all' target
all:: test/$1
.PHONY: test/$1 

endef

#-------------------------------------------------------------------------------
#
# MNIST_DATA(name, url)
# 
# Macro for downloading and unpacking the MNIST dataset
# 
define GET_MNIST 

$$(DAT_DIR)/$1.gz: | $$(DAT_DIR)
	# Retrieving $2/$1.gz 
	curl $2/$1.gz > $$@

$$(DAT_DIR)/$1: $$(DAT_DIR)/$1.gz
	# Unpacking $$(DAT_DIR)/$1.gz
	gunzip -c $$< > $$@

mnist:: $$(DAT_DIR)/$1
endef

#-------------------------------------------------------------------------------
#
# BUILD_DIR(name)
# 
# Macro for creating targets for build directories (bin, obj, dep for example).
# Typically these folders aren't in version control.
# 
define BUILD_DIR 

$1:
	mkdir $1

endef

#-------------------------------------------------------------------------------
#
# Generate MNIST targets
$(eval $(foreach mnist,$(MNIST_DATA),$(call GET_MNIST,$(mnist),$(MNIST_URL))))

#-------------------------------------------------------------------------------
#
# Generate BUILD_DIR targets
$(eval $(foreach bdir,$(BUILD_DIRS),$(call BUILD_DIR,$(bdir))))

#-------------------------------------------------------------------------------
#
# Some test cases.  Who needs a framework?
#
$(eval $(call TEST_CASE,feedforwardnetwork1,$(TST_DIR)/FeedForwardNetworkTest1.cpp,,mnist))
