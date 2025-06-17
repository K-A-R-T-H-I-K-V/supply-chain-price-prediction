// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13;

contract PredictionLogger {
    struct PredictionRecord {
        string fileHash; // Hash of the predictions file
        uint256 timestamp; // Timestamp of logging
        uint256 rmse; // RMSE of the model (scaled to integer)
        uint256 r2; // R² score (scaled to integer, e.g., 85 for 0.85)
        address logger; // Address that logged the record
    }

    PredictionRecord[] public records;

    event PredictionLogged(
        string fileHash,
        uint256 timestamp,
        uint256 rmse,
        uint256 r2,
        address indexed logger
    );

    function logPrediction(
        string memory _fileHash,
        uint256 _rmse,
        uint256 _r2
    ) public {
        records.push(
            PredictionRecord({
                fileHash: _fileHash,
                timestamp: block.timestamp,
                rmse: _rmse,
                r2: _r2,
                logger: msg.sender
            })
        );
        emit PredictionLogged(_fileHash, block.timestamp, _rmse, _r2, msg.sender);
    }

    function getRecordCount() public view returns (uint256) {
        return records.length;
    }

    function getRecord(uint256 index) public view returns (
        string memory fileHash,
        uint256 timestamp,
        uint256 rmse,
        uint256 r2,
        address logger
    ) {
        require(index < records.length, "Index out of bounds");
        PredictionRecord memory record = records[index];
        return (record.fileHash, record.timestamp, record.rmse, record.r2, record.logger);
    }
}