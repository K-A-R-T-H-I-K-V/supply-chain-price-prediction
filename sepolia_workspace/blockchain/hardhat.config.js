require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

module.exports = {
  solidity: "0.8.13",
  networks: {
    sepolia: {
      url: process.env.SEPOLIA_RPC_URL || "http://geth:8545",
      accounts: [process.env.SEPOLIA_PRIVATE_KEY],
    },
  },
};