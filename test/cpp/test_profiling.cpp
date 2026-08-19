#include <gtest/gtest.h>
#include <knexus.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#define SUCCESS 0
#define FAILURE 1

int g_argc;
char** g_argv;

int test_profiling(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0]
              << " <runtime_name> <kernel_file> <kernel_name>" << std::endl;
    return FAILURE;
  }

  std::string runtime_name = argv[1];
  std::string kernel_file = argv[2];
  std::string kernel_name = argv[3];

  auto sys = knexus::getSystem();
  auto runtime = sys.getRuntime(runtime_name);
  if (!runtime) {
    std::cout << "No runtimes found" << std::endl;
    return FAILURE;
  }

  auto devices = runtime.getDevices();
  if (devices.empty()) {
    std::cout << "No devices found" << std::endl;
    return FAILURE;
  }

  knexus::Device dev0 = runtime.getDevice(0);

  size_t vsize = 1024;
  std::vector<float> vecA(vsize, 1.0);
  std::vector<float> vecB(vsize, 2.0);
  std::vector<float> vecResult_GPU(vsize, 0.0);
  size_t size = vsize * sizeof(float);

  auto nlib = dev0.createLibrary(kernel_file);
  auto kern = nlib.getKernel(kernel_name);
  if (!kern) return FAILURE;

  auto buf0 = dev0.createBuffer(size, vecA.data());
  auto buf1 = dev0.createBuffer(size, vecB.data());
  auto buf2 = dev0.createBuffer(size, vecResult_GPU.data());

  // Profiling-enabled stream: every schedule run on this stream gets an
  // NVTX range and is captured by `nsys profile --capture-range=cudaProfilerApi`.
  auto stream0 = dev0.createStream(NXS_StreamSettings_Profiling);

  const int kNumRuns = 20;
  for (int i = 0; i < kNumRuns; ++i) {
    auto sched = dev0.createSchedule();
    auto cmd = sched.createCommand(kern);
    cmd.setArgument(0, buf0);
    cmd.setArgument(1, buf1);
    cmd.setArgument(2, buf2);
    cmd.finalize({32, 1, 1}, {32, 1, 1}, 0);

    sched.run(stream0, NXS_ExecutionSettings_Timing);
    auto time_ms = sched.getProp<nxs_double>(NP_ElapsedTime);
    std::cout << "run " << i << " elapsed_ms=" << time_ms << std::endl;
  }

  buf2.copy(vecResult_GPU.data(), NXS_BufferDeviceToHost);

  for (auto v : vecResult_GPU) {
    if (v != 3.0) {
      std::cout << "Fail: unexpected result " << v << std::endl;
      return FAILURE;
    }
  }

  std::cout << std::endl << "Test PASSED" << std::endl << std::endl;
  return SUCCESS;
}

class KNexusIntegration : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(KNexusIntegration, PROFILING) {
  int result = test_profiling(g_argc, g_argv);
  EXPECT_EQ(result, SUCCESS);
}

int main(int argc, char** argv) {
  g_argc = argc;
  g_argv = argv;

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
