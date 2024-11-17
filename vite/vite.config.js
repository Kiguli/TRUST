import { defineConfig } from "vite";
import { join, resolve } from "path";
import vue from "@vitejs/plugin-vue";
import colors from "picocolors";
import fs from "fs";

const flask = () => ({
    configureServer (server) {
        const envDir = join(process.cwd(), "..");
        const appUrl = fs.readFileSync(join(envDir, ".env"), "utf-8").
            split("\n").
            find((line) => line.startsWith("APP_URL")).
            split("=")[1];

        const flaskVersion = () => {
            try {
                const requirements = fs.readFileSync(
                    join(envDir, "requirements.txt"), "utf-8");
                const match = requirements.match(/Flask==(.+)/);
                return match ? match[1] : "unknown";
            } catch {
                return "";
            }
        };

        setTimeout(() => {
            server.config.logger.info(`\n  ${colors.red(
                `${colors.bold("FLASK")} ${flaskVersion()}`)}`);
            server.config.logger.info("");
            server.config.logger.info(
                `  ${colors.green("➜")}  ${colors.bold(
                    "APP_URL")}: ${colors.cyan(
                    appUrl.replace(/:(\d+)/,
                        (_, port) => `:${colors.bold(port)}`))}`);
        }, 200);
    },
});

export default defineConfig({
    base: "./",
    plugins: [
        vue({
            template: {
                transformAssetUrls: {
                    base: null,
                    includeAbsolute: false,
                },
            },
        }),
        flask(),
    ],
    resolve: {
        alias: {
            "@": resolve("./js/Components"),
            "~": resolve("./js"),
        },
    },
    server: {
        port: 3000,
        strictPort: true,
        host: true,
        origin: "http://0.0.0.0:3000"
    },
});
