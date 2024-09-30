<script setup>
defineProps({
    hasError: {
        type: Boolean,
        default: false,
    },
})

const model = defineModel();

</script>

<template>
    <input
        v-model="model"
        :class="[
            hasError ? 'ring-2 ring-inset ring-red-600 dark:ring-2 focus:ring-red-600' : '',
            'block h-10 w-full min-w-0 flex-1 rounded-r-md border-0 px-2 py-1.5',
            'text-sm leading-6 text-gray-900 outline-none ring-1 ring-inset ring-gray-300 ',
            'placeholder:text-gray-400',
            'focus:ring-2 focus:ring-inset focus:ring-purple-300',
            'dark:border-none dark:bg-gray-950 dark:text-gray-100 dark:outline-none dark:ring-0 disabled:opacity-30'
        ]" />
</template>
