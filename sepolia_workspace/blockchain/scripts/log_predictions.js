const { ethers } = require("ethers");
const fs = require("fs");
const keccak256 = require("keccak256");
const PredictionLoggerABI = require("../artifacts/contracts/PredictionLogger.sol/PredictionLogger.json");
require("dotenv").config();

async function main() {
  // Connect to Sepolia
  const provider = new ethers.JsonRpcProvider(process.env.SEPOLIA_RPC_URL || "http://geth:8545");
  const wallet = new ethers.Wallet(process.env.SEPOLIA_PRIVATE_KEY, provider);

  // Load the contract using the address from .env
  const contractAddress = process.env.CONTRACT_ADDRESS;
  if (!contractAddress) {
    throw new Error("CONTRACT_ADDRESS not set in .env");
  }
  const predictionLogger = new ethers.Contract(contractAddress, PredictionLoggerABI.abi, wallet);

  // Read the predictions file from the blockchain directory
  const filePath = "./test_predictions.csv";
  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found at ${filePath}. Please copy test_predictions.csv to the blockchain directory.`);
  }
  const fileData = fs.readFileSync(filePath);
  const fileHash = "0x" + keccak256(fileData).toString("hex");

  // Hardcode RMSE and R² from Step 2 (Random Forest)
  const rmse = 2630; // $26.30 * 100 (scaled to integer)
  const r2 = 85; // 0.85 * 100 (scaled to integer)

  // Log the prediction
  const tx = await predictionLogger.logPrediction(fileHash, rmse, r2);
  await tx.wait();
  console.log("Prediction logged! Transaction hash:", tx.hash);

  // Verify the record
  const recordCount = await predictionLogger.getRecordCount();
  const record = await predictionLogger.getRecord(recordCount - 1n); // Use BigInt for recordCount
  console.log("Logged Record:", {
    fileHash: record[0],
    timestamp: record[1].toString(), // Already a string, but ensuring consistency
    rmse: record[2].toString(), // Convert BigInt to string
    r2: record[3].toString(), // Convert BigInt to string
    logger: record[4],
  });
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});