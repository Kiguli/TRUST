<script setup></script>

<template>
    <div
        class="flex h-56 flex-col items-center rounded-lg border-2 border-dashed border-gray-300 p-4 focus:border-blue-300 focus:outline-none dark:border-gray-600">
        <div class="py-12 text-center text-sm">
            <p class="mb-1 font-medium text-gray-900 dark:text-gray-200">
                Choose a file.
            </p>
            <p class="text-gray-400">
                CSV, TXT, JSON, and ZIP files, up to 50MB.
            </p>
        </div>

        <input
            :class="[
                'file:m-1 file:mr-4 file:cursor-pointer file:rounded-md file:border-none file:bg-violet-900/75 file:px-5 file:py-3 file:text-white',
                'file:hover:bg-violet-900/80 file:focus:ring-2 file:focus:ring-white/60 file:focus:ring-offset-2 file:focus:ring-offset-violet-300',
                'file:active:ring-2 file:active:ring-white/60 file:active:ring-offset-2 file:active:ring-offset-violet-300',
            ]"
            class="block w-full rounded-lg border border-gray-300 bg-gray-100 text-sm text-gray-500 focus:outline-none dark:border-none dark:bg-gray-950"
            type="file"
            @input="dataset = $event.target.files[0]" />
        <progress v-if="progress" :value="progress.percentage" max="100">
            {{ progress.percentage }}
        </progress>
    </div>
</template>

<style scoped></style>
