FROM node:20

WORKDIR /app

COPY blockchain/package.json blockchain/package-lock.json ./
RUN npm install

COPY blockchain/ .

CMD ["npx", "hardhat", "run", "scripts/deploy.js", "--network", "sepolia"]