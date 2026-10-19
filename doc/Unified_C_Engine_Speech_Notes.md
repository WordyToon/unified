# AI Accelerator Design & Development — Presenter Speech Notes

Unified C-Engine · AMD ZCU106 · 17 slides

Suggested delivery: approximately 12–15 minutes. Narration is also embedded in the PowerPoint Notes pane and Presenter View.

## Slide 01 — AI Accelerator Design & Development

This presentation introduces the design and development of our Unified C-Engine AI accelerator on the AMD ZCU106. The aim is to build a programmable compute platform that can grow over time. Four model integrations are available today: MNIST LeNet-5, ResNet-50, MobileNetV2 and EfficientNet-B0. More models and operators will follow as the compiler and RTL evolve. I will explain the system and engine architecture, the memory hierarchy, and the software flow, then show the implementation evidence and the remaining development milestones. The results are drawn from saved repository artifacts, so each performance claim is tied to its recorded scope.

## Slide 02 — An AI accelerator platform with four model integrations

The project is organized around one reusable, 256-processing-element fabric and two precision builds. Model-specific weights and microcode configure the engine for different workloads. The four current integrations cover digit recognition, residual networks, depthwise convolution and MBConv-style vision processing. The instruction counts on this slide describe the generated programs currently stored in the repository. Support and validation are separate milestones: a model can have a compiler flow while still needing stronger numerical or board validation. The long-term direction is to keep expanding coverage through compiler mappings, operator implementations and a consistent verification process.

## Slide 03 — From model assets to FPGA inference

At the system level, the ARM processor manages the inference job. It prepares configuration, launches transfers and presents the returned predictions. The FPGA programmable logic executes the scheduled neural compute. Images and weights begin in DDR, move through the DMA read path into banked SRAM, and feed the compute array. Post-processing transforms the accumulated results before they return to SRAM. The write DMA then returns the final logits to DDR. Microcode coordinates the layer sequence. This separation provides a reusable accelerator architecture while keeping control and result handling in a familiar baremetal software environment.

## Slide 04 — FPGA implementation: the PS–PL system diagram

This diagram maps the accelerator onto the actual Vivado system connections. At the top, the ARM processing system sends configuration through its HPM master port and a control SmartConnect to the accelerator's AXI-Lite interface. The accelerator has separate read and write DMA masters. Read data returns from DDR through HP0, while output data travels through HP1 on the write path. The arrows here show payload movement; the DMA requests themselves originate in the FPGA logic. The design shares the PS-generated PL clock, uses the processor-system reset block, and connects the accelerator interrupt back to the PS. These interfaces form the system-level contract for future accelerator extensions.

## Slide 05 — RTL engine architecture: modules and data paths

Here we move inside the RTL hierarchy. The outer DMA wrapper contains the AXI-Lite register interface, the read and write masters, and the bridge that maps transfer beats into SRAM banks. Inside the compute core, configuration memory feeds the sequencer. The controller selects operands and controls the processing-element array. Array results pass through the requantization and post-processing unit, which is instantiated inside the controller, before returning to SRAM. Port A is shared through a DMA-versus-controller multiplexer, while Port B serves the compute path. This modular boundary helps us identify what changes when adding an operator: its scheduling, operand steering, arithmetic or post-processing requirements.

## Slide 06 — Inside the compute engine

The compute fabric contains 256 signed multiply–accumulate units arranged as eight rows by thirty-two columns. Each processing element can use local accumulation and horizontal or vertical forwarding, giving the controller several ways to map computation onto the grid. Accumulator precision is wider than input precision: thirty-two bits for the INT8 configuration and forty-eight bits for INT16. After accumulation, the output passes through shift-based requantization, saturation, residual processing and the configured activation or pooling path. The 38.4 GOP/s figure is a theoretical arithmetic ceiling at 75 MHz. Actual model throughput also depends on utilization, scheduling and data movement.

## Slide 07 — The software makes the hardware reusable

The compiler is what makes the hardware useful across different models. It imports model parameters, folds batch normalization, quantizes the numerical representation and builds a schedule for tensors and operations. The generated outputs include weight binaries, microcode and simulation material. During deployment, JTAG and XSDB load the assets, the ARM driver launches the accelerator, and logits return for display over UART. One evaluation detail matters: the current ImageNet compiler includes a sixty-dimensional classification-head projection calibrated using test imagery. This makes held-out accuracy evaluation an important next step as we expand the platform beyond the current examples.

## Slide 08 — Memory hierarchy: DDR capacity to PE-local reuse

The memory hierarchy connects external capacity to local reuse. DDR holds large model assets, input images and returned logits. DMA transfers use sixteen-byte beats, while banked SRAM presents a much wider row to the compute core. The controller stages weights and holds intermediate pooling and output values close to the processing elements. Finally, each PE has local and forwarding registers to retain partial results. This is explicitly managed storage rather than an automatic cache hierarchy. A separate four-kibibyte instruction store controls the schedule. Supporting more models therefore requires both suitable operators and a memory plan that fits the tensor sizes and transfer patterns.

## Slide 09 — Memory planning is central to scaling

This slide compares the two memory configurations. INT8 uses sixteen banks and offers about 3.93 megabytes of logical tensor storage. INT16 doubles the row width and uses a smaller depth, giving about 2.10 megabytes. Both configurations provide a separate store for 256 microcode instructions. The lower diagram shows an example allocation from the current ResNet-50 compiler: input and early-stage storage, two alternating intermediate buffers, a residual buffer and a weight staging region. The allocation is part of the compiler contract. As new models and operators are added, their tensor sizes and lifetimes must be checked against these available regions.

## Slide 10 — Both saved hardware builds meet timing

The saved routed reports provide the physical baseline for the design. The INT8 export meets timing at approximately 75 MHz with positive setup slack. The INT16 export also meets timing; its actual reported clock is about 68.176 MHz, even though the configuration requests 70 MHz. The resource chart shows that INT8 uses all available UltraRAM blocks, while the INT16 build has a larger logic footprint and a different memory balance. These figures are useful constraints for further development. Any additional operator or larger buffer must be assessed for resource cost and timing impact, and the updated build needs its own routed evidence.

## Slide 11 — Recorded simulation results: three completed runs

The saved INT8 simulation logs contain three completed model runs. MNIST completes its recorded compact-input path with ten matching logits. MobileNetV2 and EfficientNet-B0 record full RGB network paths and zero mismatches against their logged quantized references. Their cycle counts correspond to roughly 5.6 and 7.7 seconds at 75 MHz. These are calculated durations for the recorded compute interval, excluding host setup. The saved ResNet-50 trace reaches 134 million cycles but has no completion marker. It remains part of the supported development scope, with completion of full-network validation still an important milestone. These logs were reviewed, not rerun for this presentation.

## Slide 12 — A correct top-1 example is only part of accuracy

This example helps explain why we look beyond the pass marker. The dog-image runs report Samoyed as top one. MobileNetV2 has a clear highest logit in this capture. EfficientNet-B0, however, gives three classes the maximum INT8 value of 127, so its top result includes a saturation tie. Agreement with a quantized reference is valuable evidence about execution, but it does not establish general classification accuracy. We need to examine output distributions, fix saturation where appropriate, and evaluate on images that were not used for calibration. That becomes more important as we bring additional models onto the accelerator.

## Slide 13 — The board flow runs; output quality needs closure

The physical-board capture confirms that the deployment chain can boot the baremetal application, execute the accelerator and read logits back. The saved MNIST run reports 1,685 hardware cycles and approximately 22.46 microseconds. It also contains four tied maximum logits, which means successful execution alone does not close numerical validation. This board run is a separate recorded snapshot from the simulation results. The next acceptance step is to align the bitstream, microcode and model assets, compare the board logits with a fixed-point reference, and repeat the capture for all four current model integrations.

## Slide 14 — The next revision targets timing and data movement

The next hardware revision focuses on timing and data movement. A staged global-average-pooling reduction is already present in the reviewed RTL. The V4 documents also describe a larger ping-pong buffering plan and overlapping weight transfers with computation. They propose a 100 MHz operating point and a ResNet-50 throughput budget near 4.9 frames per second. Those numbers should be treated as targets until a matching routed export and complete execution measurements are available. The engineering task is to connect the architectural improvements to a versioned build and demonstrate their effect on real inference performance.

## Slide 15 — A complete build-to-board toolchain is available

The repository includes a build-to-board workflow rather than only an isolated compute block. The numbered scripts compile model assets, package the accelerator IP, build the FPGA image, build the baremetal runtime, run simulation and deploy to the board. This workflow provides a practical basis for adding further models and operators. Each addition should produce compatible model assets and integrate into the same validation sequence. The command at the bottom is an example of the existing deployment entry point. For a release, we should preserve the exact source version, configuration and generated artifacts alongside its logs so another engineer can reproduce the result.

## Slide 16 — Development roadmap: more models, more operators

The development roadmap has two connected goals: increase capability and improve confidence in that capability. Today we have a programmable 256-PE fabric, four model integration flows, several common neural-network operators, dual precision and an integrated toolchain. Next, we plan to bring more models into the compiler and extend the RTL and microcode where additional operators are required. In parallel, we need to complete the outstanding ResNet validation, resolve numerical issues, evaluate held-out accuracy and measure performance on hardware. Each new capability should progress through operator checks, model validation, timing closure and a board demonstration before it becomes a validated release feature.

## Slide 17 — Evidence guide and important distinctions

This appendix shows where the presentation evidence comes from. The architecture is based on RTL and compiler files. Physical clock and resource figures come from routed reports. Functional outcomes come from saved simulation logs, and the board status comes from the UART capture. The V4 documents are used to explain the development targets. The distinction matters: theoretical peak arithmetic is different from sustained throughput, and a successful example is different from a dataset accuracy result. Detailed source notes remain attached to every slide so the design and results can be followed back to their repository evidence.
