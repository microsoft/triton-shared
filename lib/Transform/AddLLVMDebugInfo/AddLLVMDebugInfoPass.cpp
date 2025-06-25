//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton-shared/Transform/AddLLVMDebugInfo/AddLLVMDebugInfo.h"

#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Path.h"

#define DEBUG_TYPE "add-llvm-debug-info"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Transform/AddLLVMDebugInfo/Passes.h.inc"

namespace {

class AddLLVMDebugInfoPass : public AddLLVMDebugInfoBase<AddLLVMDebugInfoPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::MLIRContext *context = &getContext();
    mlir::Builder builder(context);
    mlir::SymbolTable symbolTable(moduleOp);

    // iterate through all functions
    moduleOp.walk([&](triton::FuncOp funcOp) {
      mlir::StringRef fileName;
      mlir::StringRef filePath;
      unsigned line;
      unsigned col;
      bool isOptimized = funcOp->hasAttr("llvm.optimized");
      mlir::LLVM::DIEmissionKind emissionKind = mlir::LLVM::DIEmissionKind::LineTablesOnly;

      // get loc for function and pull line and file information
      mlir::Location loc = funcOp.getLoc();
      if (auto funcLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
        fileName = llvm::sys::path::filename(funcLoc.getFilename().getValue());
        filePath = llvm::sys::path::parent_path(funcLoc.getFilename().getValue());
        line = funcLoc.getLine();
        col = funcLoc.getColumn();
      }

      // initialize useful attributes
      mlir::LLVM::DIFileAttr fileAttr = mlir::LLVM::DIFileAttr::get(context, fileName, filePath);
      mlir::StringAttr producer = mlir::StringAttr::get(context, "MLIR");
      mlir::LLVM::DICompileUnitAttr cuAttr = mlir::LLVM::DICompileUnitAttr::get(
        mlir::DistinctAttr::create(mlir::UnitAttr::get(context)),
        llvm::dwarf::getLanguage("DW_LANG_Python"), fileAttr, producer,
        isOptimized, emissionKind);

      // get subroutine types
      llvm::SmallVector<mlir::LLVM::DITypeAttr> types;

      // create subroutine type attribute from return and argument types
      unsigned callingConvention = llvm::dwarf::DW_CC_normal;
      mlir::LLVM::DISubroutineTypeAttr type = mlir::LLVM::DISubroutineTypeAttr::get(context, callingConvention, types);
      
      // set flags
      mlir::LLVM::DISubprogramFlags subprogramFlags = mlir::LLVM::DISubprogramFlags{};
      if (!funcOp.isDeclaration()) {
        subprogramFlags = mlir::LLVM::DISubprogramFlags::Definition;
      }
      if (isOptimized) {
        subprogramFlags = subprogramFlags | mlir::LLVM::DISubprogramFlags::Optimized;
      }
      if (funcOp.getSymNameAttr() == "main") {
        subprogramFlags = subprogramFlags | mlir::LLVM::DISubprogramFlags::MainSubprogram;
      }
      
      // retained nodes
      llvm::ArrayRef<mlir::LLVM::DINodeAttr> importedModules;

      // annotations
      llvm::ArrayRef<mlir::LLVM::DINodeAttr> annotations;

      // initialize DI attribute for function
      mlir::LLVM::DISubprogramAttr spAttr = mlir::LLVM::DISubprogramAttr::get(
        context,
        mlir::DistinctAttr::create(mlir::UnitAttr::get(context)),
        cuAttr,
        fileAttr, // scope
        funcOp.getSymNameAttr(),
        funcOp.getSymNameAttr(), // linkage name
        fileAttr,
        line,
        col, // scope line
        subprogramFlags,
        type,
        importedModules,
        annotations
      );
      
      // annotate function
      funcOp->setLoc(builder.getFusedLoc({loc}, spAttr));
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createAddLLVMDebugInfoPass() {
  return std::make_unique<AddLLVMDebugInfoPass>();
}
