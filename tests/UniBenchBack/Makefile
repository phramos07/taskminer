CC=clang

PLATFORM_MK=./common/c.mk
include $(PLATFORM_MK)

BENCH_MK=$(BENCH_DIR)/../Makefile
include $(BENCH_MK)

SRC_MK=$(BENCH_DIR)/src/Makefile
include $(SRC_MK)

$(BENCH_DIR)/build/$(BENCH_NAME):
	make compile

$(BENCH_DIR)/build:
	mkdir $(BENCH_DIR)/build

$(BENCH_DIR)/log:
	mkdir $(BENCH_DIR)/log

cleanbin:
	rm -rf $(BENCH_DIR)/build/*

cleanlog:
	rm -rf $(BENCH_DIR)/log/*

cleanall: cleanbin cleanlog

compile: $(BENCH_DIR)/build $(BENCH_DIR)/log
	echo "Compiling" $(BENCH_NAME); \
	echo "\n---------------------------------------------------------" >> $(BENCH_DIR)/log/$(BENCH_NAME).compile; \
	date >> $(BENCH_DIR)/log/$(BENCH_NAME).compile; \
	echo "$(CC) $(COMMON_FLAGS) $(BENCH_FLAGS) $(AUX_SRC) $(SRC_OBJS) -o $(BENCH_DIR)/build/$(BENCH_NAME)" 2>> $(BENCH_DIR)/log/$(NAME).compile; \
	$(CC) $(COMMON_FLAGS) $(BENCH_FLAGS) $(AUX_SRC) $(SRC_OBJS) -o $(BENCH_DIR)/build/$(BENCH_NAME) 2>> $(BENCH_DIR)/log/$(BENCH_NAME).compile; \
	rm -f _kernel*.cl~
	mv _kernel* $(BENCH_DIR)/build/; \
	echo ""

run: $(BENCH_DIR)/build/$(BENCH_NAME)
	cd $(BENCH_DIR)/build;\
	echo "Running" $(BENCH_NAME); \
	echo "./$(BENCH_NAME) $(INPUT_FLAGS)" >> ../log/$(BENCH_NAME).execute; \
<<<<<<< HEAD
	./$(BENCH_NAME) $(INPUT_FLAGS) >> ../log/$(BENCH_NAME).tmp; \
=======
	./$(BENCH_NAME) $(INPUT_FLAGS) > ../log/$(BENCH_NAME).tmp; \
	cat ../log/$(BENCH_NAME).tmp; \
	cat ../log/$(BENCH_NAME).tmp >> ../log/$(BENCH_NAME).execute; \
	rm ../log/$(BENCH_NAME).tmp; \
	echo "\n"

runmali: $(BENCH_DIR)/build/$(BENCH_NAME)
	cd $(BENCH_DIR)/build;\
	echo "Copying files to device"; \
	adb shell "mkdir -p /data/local/tmp/$(BENCH_DIR)"; \
	adb push ../build /data/local/tmp/$(BENCH_DIR)/build; \
	adb push ../input /data/local/tmp/$(BENCH_DIR)/input; \
	echo "\nRunning" $(BENCH_NAME); \
	echo "./$(BENCH_NAME) $(INPUT_FLAGS)" >> ../log/$(BENCH_NAME).execute; \
	adb shell "cd /data/local/tmp/$(BENCH_DIR)/build; ./$(BENCH_NAME) $(INPUT_FLAGS) > $(BENCH_NAME).tmp"; \
	adb pull "/data/local/tmp/$(BENCH_DIR)/build/$(BENCH_NAME).tmp" ../log/$(BENCH_NAME).tmp; \
>>>>>>> 30dea79fa86a80dcc9d6aaebaf5a6bc5ecfeafc8
	cat ../log/$(BENCH_NAME).tmp; \
	cat ../log/$(BENCH_NAME).tmp >> ../log/$(BENCH_NAME).execute; \
	rm ../log/$(BENCH_NAME).tmp; \
	echo "\n"
