### Added

- **A GGUF the installed llama.cpp cannot load says whether updating llama.cpp fixes it.** AbstractCore ships the
  architecture table of the newest llama-cpp-python release (`assets/llamacpp_known_architectures.json`: 0.3.36,
  153 architectures from the vendored llama.cpp's `LLM_ARCH_NAMES`, spark2_5 included). An installed or catalog
  GGUF whose architecture the installed llama.cpp does not know but 0.3.36 does carries `llamacpp_update`
  (`{architecture, installed, update_to}`) and the sentence "... cannot load the spark2_5 architecture;
  llama-cpp-python 0.3.36 can: update llama.cpp." (the console offers "Update llama.cpp"); only an architecture no
  release knows stays greyed ("No llama.cpp release supports the X architecture yet"). Every catalog GGUF's Hub
  architecture (86 repos, 28 architectures, checked 2026-10-29) is in the 0.3.36 table.
