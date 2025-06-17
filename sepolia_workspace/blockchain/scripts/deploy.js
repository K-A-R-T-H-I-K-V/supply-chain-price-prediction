const hre = require("hardhat");

async function main() {
    const PredictionLogger = await hre.ethers.getContractFactory("PredictionLogger");
    const predictionLogger = await PredictionLogger.deploy();
    await predictionLogger.waitForDeployment();
    console.log("PredictionLogger deployed to:", await predictionLogger.getAddress());
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});