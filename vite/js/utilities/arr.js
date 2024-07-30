const wrap = (value) => {
    if (value === null || value === undefined) {
        return [];
    }

    return Array.isArray(value) ? value : [value];
}

export { wrap };